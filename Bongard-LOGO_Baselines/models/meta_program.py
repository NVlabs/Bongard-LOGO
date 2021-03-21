# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register


@register('meta-program')
class MetaProgram(nn.Module):

    def __init__(self, prog_synthesis, prog_synthesis_args={}, prog_use_mode=0, update_prog_synthesis=True,
                 method='cos', temp=10., temp_learnable=True):
        """
        Meta Learning with pretrained program synthesis
        :param prog_synthesis: the pre-trained program synthesis module
        :param prog_synthesis_args: the arguments of program synthesis module
        :param prog_use_mode: 0 - image feature only, 1 - program feature only, 2 - program only,
                              3 - image feature & program feature, 4 - image feature & program
        """
        super().__init__()
        self.prog_synthesis = models.make(prog_synthesis, **prog_synthesis_args)
        self.prog_use_mode = prog_use_mode
        self.update_prog_synthesis = update_prog_synthesis
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

        print('prog_use_mode: {}'.format(self.prog_use_mode))

        if self.prog_use_mode != 0:
            if self.prog_use_mode in [1, 2]:
                emb_dim = self.prog_synthesis.n_layers * self.prog_synthesis.hidden_dim
            else:
                if self.prog_synthesis.recurrent_model.model_type == 'lstm':
                    emb_dim = self.prog_synthesis.encoder.out_dim + \
                              self.prog_synthesis.n_layers * self.prog_synthesis.hidden_dim
                else:
                    emb_dim = self.prog_synthesis.encoder.out_dim + self.prog_synthesis.hidden_dim
            self.out_layer = nn.Linear(emb_dim, self.prog_synthesis.encoder.out_dim)

    def forward(self, x_shot, x_query, **kwargs):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)

        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))

        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)  # [ep_per_batch, way, feature_len]
            x_query = F.normalize(x_query, dim=-1)  # [ep_per_batch, way * query, feature_len]
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
            x_query, x_shot, metric=metric, temp=self.temp)  # [ep_per_batch, way * query, way]
        return logits

    def encoder(self, images):
        batch_size = images.size(0)
        img_feat = self.prog_synthesis.encoder(images)

        if self.prog_use_mode == 0:  # image feature only
            if not self.update_prog_synthesis:
                img_feat = img_feat.detach()
            encoded_feat = img_feat  # [bs, out_dim]

        else:
            sampled_seq, hidden = self.prog_synthesis.sample(images)
            assert isinstance(hidden, torch.Tensor)

            if self.prog_use_mode == 1:  # program feature only

                if self.prog_synthesis.recurrent_model.model_type == 'lstm':
                    hidden_feat = hidden.transpose(0, 1).contiguous().view(batch_size, -1)  # [bs, nlayers*hiddim]
                if not self.update_prog_synthesis:
                    hidden_feat = hidden_feat.detach()
                encoded_feat = self.out_layer(hidden_feat)

            elif self.prog_use_mode == 2:  # program only

                sample_feat = self.sampled_seq_encoder(sampled_seq.detach())  # [nlayers, bs, hiddim]
                if self.prog_synthesis.recurrent_model.model_type == 'lstm':
                    sample_feat = sample_feat.transpose(0, 1).contiguous().view(batch_size, -1)  # [bs, nlayers*hiddim]
                if not self.update_prog_synthesis:
                    sample_feat = sample_feat.detach()
                encoded_feat = self.out_layer(sample_feat)

            elif self.prog_use_mode == 3:  # image feature + program feature

                if self.prog_synthesis.recurrent_model.model_type == 'lstm':
                    hidden_feat = hidden.transpose(0, 1).contiguous().view(batch_size, -1)  # [bs, nlayers*hiddim]
                combined_feat = torch.cat([img_feat, hidden_feat], dim=-1)  # [bs, out_dim + nlayers * hiddim]
                if not self.update_prog_synthesis:
                    combined_feat = combined_feat.detach()
                encoded_feat = self.out_layer(combined_feat)

            elif self.prog_use_mode == 4:  # image feature + program feature

                sample_feat = self.sampled_seq_encoder(sampled_seq.detach())  # [nlayers, bs, hiddim]
                if self.prog_synthesis.recurrent_model.model_type == 'lstm':
                    sample_feat = sample_feat.transpose(0, 1).contiguous().view(batch_size, -1)  # [bs, nlayers*hiddim]
                combined_feat = torch.cat([img_feat, sample_feat], dim=-1)  # [bs, out_dim + nlayers*hiddim]
                if not self.update_prog_synthesis:
                    combined_feat = combined_feat.detach()
                encoded_feat = self.out_layer(combined_feat)

            else:
                raise Exception('Unkown prog_use_mode: {}!'.format(self.prog_use_mode))

        return encoded_feat

    def sampled_seq_encoder(self, sampled_seq):
        # bs x seq_len x 4 -> seq_len x bs x 4
        sampled_seq = torch.transpose(sampled_seq, 0, 1).contiguous()

        sampled_seq = sampled_seq[:-1, :, :]  # no last token, [seq_len - 1, bs, 4]
        batch_size = sampled_seq.size(1)

        emb_seq = self.prog_synthesis.program2embs(sampled_seq)  # [seq_len - 1, bs, emb_dim]

        if self.prog_synthesis.recurrent_model.model_type == 'lstm':
            self.prog_synthesis.recurrent_model.flatten_parameters()
            hidden = self.prog_synthesis.recurrent_model.init_hidden(batch_size)
            _, hidden = self.prog_synthesis.recurrent_model(emb_seq, hidden)
            hidden = hidden[0]  # only use the first hidden

        else:
            _, hidden = self.prog_synthesis.recurrent_model(emb_seq)  # todo: use 1D convolution to replace lstm encoder

        assert isinstance(hidden, torch.Tensor)

        return hidden
