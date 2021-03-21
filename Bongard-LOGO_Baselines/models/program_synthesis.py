# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from .models import register
from datasets.shape_program import BASE_ACTIONS, BASE_LINE_TYPES, MAX_LEN_PROGRM
import utils


@register('program-decoder')
class ProgramDecoder(nn.Module):
    def __init__(self, encoder, encoder_args={}, recurrent_model='lstm', seq_len=MAX_LEN_PROGRM, embed_dim=64,
                 hidden_dim=64, n_layers=2, drop_prob=0.5, continuous=True, num_head=2,
                 always_with_input_img=True, use_mixture_density=True, components_size=5,
                 discretized_num=10, base_type_coef=1.0, arg_coef=1.0, repackage=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.recurrent_model = models.make(recurrent_model, nemb=embed_dim, nhead=num_head,
                                           nhid=hidden_dim, nlayers=n_layers, dropout=drop_prob,
                                           repackage=repackage)
        self.feat_len = self.encoder.out_dim
        self.continuous = continuous
        self.seq_len = seq_len  # 3 (start, and, stop) + 9 (max No. of actions) + 9 (max No. of actions)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.always_with_input_img = always_with_input_img
        self.use_mixture_density = use_mixture_density
        self.discretized_num = discretized_num
        self.base_type_coef = base_type_coef
        self.arg_coef = arg_coef

        self.base_decoder_names = BASE_ACTIONS
        self.base_decoder_types = BASE_LINE_TYPES

        print('recurrent_model type: ', self.recurrent_model.model_type)
        print('continuous: ', self.continuous)
        print('use_mixture_density: ', self.use_mixture_density)

        self.primitive_decoder = PrimitiveDecoder(self.hidden_dim, self.continuous, self.use_mixture_density,
                                                  components_size, self.base_decoder_names, self.base_decoder_types,
                                                  self.discretized_num, self.base_type_coef, self.arg_coef).cuda()

        self.token_len = sum(out[1] for out in self.primitive_decoder.output_dims)

        self.feat2emb = nn.Linear(self.feat_len, self.embed_dim)
        self.token2emb = nn.Linear(self.token_len, self.embed_dim)

        # self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, self.n_layers, dropout=self.drop_prob, batch_first=True)

    def forward(self, x, program):
        """ program in the form of [..., (base_idx, base_type, dist, angle), ...], shape: bs x seq_len x 4 """

        # bs x seq_len x 4 -> seq_len x bs x 4
        program = torch.transpose(program, 0, 1).contiguous()

        # discretize dist and angle if not continuous
        if not self.continuous:
            program = self.discretize_args(program)

        input_seq = program[:-1, :, :]  # (seq_len - 1) x bs x 4
        target_seq = program[1:, :, :]  # (seq_len - 1) x bs x 4

        img_feat = self.encoder(x)
        batch_size = img_feat.size(0)

        emb_seq = self.program2embs(input_seq, img_feat)  # (seq_len - 1) x bs x emb_dim

        if self.recurrent_model.model_type == 'lstm':
            self.recurrent_model.flatten_parameters()  # Resets parameter data pointer so that they can use faster code paths
            hidden = self.recurrent_model.init_hidden(batch_size)  # No need to carry hidden state across hidden state

        # def repackage_hidden(h):
        #     """Wraps hidden states in new Tensors, to detach them from their history."""
        #     if isinstance(h, torch.Tensor):
        #         return h.detach()
        #     else:
        #         return tuple(repackage_hidden(v) for v in h)
        #
        # if self.repackage:
        #     hidden = repackage_hidden(hidden)

        self.recurrent_model.zero_grad()
        if self.recurrent_model.model_type == 'lstm':
            output, hidden = self.recurrent_model(emb_seq, hidden)
        else:
            output, hidden = self.recurrent_model(emb_seq)

        assert output.size(0) == target_seq.size(0)

        target_primitives = target_seq.contiguous().view(-1, target_seq.size(-1))  # (seq_len - 1)*bs x 4
        output = output.contiguous().view(-1, self.hidden_dim)  # (seq_len - 1)*bs x hid_dim

        # create a mask by filtering out all tokens that ARE NOT the padding token
        pad_token = torch.tensor([self.base_decoder_names.index('<PAD>'), 0, 0, 0],
                                 dtype=torch.float32).cuda()
        mask = torch.any(target_primitives != pad_token, dim=-1).to(torch.float32)  # [N*T]
        # count how many effective tokens we have
        eff_tokens = float(torch.sum(mask).item())

        loss_primitive, acc_base_idx, acc_base_type, acc_args0, acc_args1 = \
            self.primitive_decoder(output, target_primitives)

        loss_ave = torch.sum(loss_primitive * mask, dim=0, keepdim=True) / eff_tokens  # a scalar tensor [1]
        acc_base_idx_ave = torch.sum(acc_base_idx * mask, dim=0, keepdim=True) / eff_tokens  # a scalar tensor [1]
        acc_base_type_ave = torch.sum(acc_base_type * mask, dim=0, keepdim=True) / eff_tokens  # a scalar tensor [1]
        acc_args0_ave = torch.sum(acc_args0 * mask, dim=0, keepdim=True) / eff_tokens  # a scalar tensor [1]
        acc_args1_ave = torch.sum(acc_args1 * mask, dim=0, keepdim=True) / eff_tokens  # a scalar tensor [1]

        return loss_ave, acc_base_idx_ave, acc_base_type_ave, acc_args0_ave, acc_args1_ave

    def sample(self, x, target_program=None):

        if target_program is not None:
            # bs x seq_len x 4 -> seq_len x bs x 4
            target_program = torch.transpose(target_program, 0, 1).contiguous()
            # discretize dist and angle if not continuous
            if not self.continuous:
                target_program = self.discretize_args(target_program)

        # inference without teacher-forcing
        with torch.no_grad():  # no need to track history in sampling
            img_feat = self.encoder(x)
            batch_size = img_feat.size(0)

            start_idx = self.base_decoder_names.index('start')
            start_primitive = torch.tensor([start_idx, 0, 0, 0]).to(torch.float32).cuda()
            start_primitive = start_primitive.expand(batch_size, -1)
            assert start_primitive.size() == (batch_size, 4)

            if self.recurrent_model.model_type == 'lstm':
                self.recurrent_model.flatten_parameters()
                hidden = self.recurrent_model.init_hidden(batch_size)

            sampled_seq = [start_primitive]
            for i in range(self.seq_len - 1):

                if self.recurrent_model.model_type == 'lstm':
                    # one-step prediction only
                    input = sampled_seq[-1].unsqueeze(dim=0)  # [1, bs, 4]
                    emb = self.program2embs(input, img_feat)  # [1, bs, emb_dim]
                    output, hidden = self.recurrent_model(emb, hidden)
                else:
                    # one-step prediction only
                    emb_prev = self.program2embs(torch.stack(sampled_seq, dim=0), img_feat)  # [seq_prev, bs, emb_dim]
                    output, hidden = self.recurrent_model(emb_prev, False)

                primitive_sample = self.primitive_decoder.sample(output[-1, :, :], i)  # [bs, 4]
                sampled_seq.append(primitive_sample)

            sampled_seq = torch.stack(sampled_seq, dim=0)  # [seq_len, bs, 4]

            if self.recurrent_model.model_type == 'lstm':
                hidden = hidden[0]  # only use the first hidden
            assert isinstance(hidden, torch.Tensor)

            # inference-time acc
            if target_program is not None:
                acc_base_idx_ave, acc_base_type_ave, acc_args0_ave, acc_args1_ave = \
                    self.calc_acc_program(sampled_seq, target_program)
                return sampled_seq.transpose(0, 1).contiguous(), hidden, \
                       acc_base_idx_ave, acc_base_type_ave, acc_args0_ave, acc_args1_ave

            return sampled_seq.transpose(0, 1).contiguous(), hidden

    # def initHidden(self, batch_size):
    #     hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(torch.float32).cuda()
    #     cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(torch.float32).cuda()
    #     hidden = (hidden_state, cell_state)
    #     return hidden

    def program2embs(self, program, img_feat=None):
        base_actions_size = len(self.base_decoder_names)
        base_act_type_size = len(self.base_decoder_types)
        emb_seq = []

        # program: [prog_len, bs, 4]
        assert program.size(0) <= self.seq_len - 1, program.size()
        if img_feat is not None:
             assert program.size(1) == img_feat.size(0), program.size()

        base_idx, base_type, args0, args1 = program[:, :, 0], program[:, :, 1], program[:, :, 2], program[:, :, 3]
        base_idx_onehot = F.one_hot(base_idx.to(torch.int64), base_actions_size).to(torch.float32)
        base_type_onehot = F.one_hot(base_type.to(torch.int64), base_act_type_size).to(torch.float32)
        if self.continuous:
            program_onehot = torch.cat([base_idx_onehot, base_type_onehot, program[:, :, 2:]], dim=-1)
        else:
            args0_onehot = F.one_hot(args0.to(torch.int64), self.discretized_num).to(torch.float32)
            args1_onehot = F.one_hot(args1.to(torch.int64), self.discretized_num).to(torch.float32)
            program_onehot = torch.cat([base_idx_onehot, base_type_onehot, args0_onehot, args1_onehot], dim=-1)
        assert program_onehot.size(-1) == self.token_len
        program_emb = self.token2emb(program_onehot)  # prog_len x bs x emb_dim

        # add image input
        if self.always_with_input_img and img_feat is not None:
            program_emb += self.feat2emb(img_feat).expand(program.size(0), -1, -1)  # prog_len x bs x emb_dim

        # for j in range(program.size(1)):
        #     primitive = program[:, j, :]
        #     base_idx, base_type, args0, args1 = primitive[:, 0], primitive[:, 1], primitive[:, 2], primitive[:, 3]
        #     base_idx_onehot = F.one_hot(base_idx.to(torch.int64), base_actions_size).to(torch.float32)
        #     base_type_onehot = F.one_hot(base_type.to(torch.int64), base_act_type_size).to(torch.float32)
        #     if self.continuous:
        #         primitive_onehot = torch.cat([base_idx_onehot, base_type_onehot, primitive[:, 2:]], dim=-1)
        #     else:
        #         args0_onehot = F.one_hot(args0.to(torch.int64), self.discretized_num).to(torch.float32)
        #         args1_onehot = F.one_hot(args1.to(torch.int64), self.discretized_num).to(torch.float32)
        #         primitive_onehot = torch.cat([base_idx_onehot, base_type_onehot, args0_onehot, args1_onehot], dim=-1)
        #     assert primitive_onehot.size(1) == self.token_len
        #     primitive_emb = self.token2emb(primitive_onehot)  # bs x emb_dim
        #
        #     # add image input
        #     if self.always_with_input_img and img_feat is not None:
        #         primitive_emb += self.feat2emb(img_feat)
        #
        #     emb_seq.append(primitive_emb)
        # emb_seq = torch.stack(emb_seq, dim=1)  # bs x (seq_len - 1) x emb_dim

        return program_emb

    def discretize_args(self, program):
        bins = torch.linspace(0, 1, self.discretized_num + 1).cuda()

        def bucketize(tensor, bucket_boundaries):
            result = torch.zeros_like(tensor, dtype=torch.float32).cuda()
            for boundary in bucket_boundaries[1:]:
                result += (tensor > boundary).to(torch.float32)
            return result

        bases, args0, args1 = program[:, :, :-2], program[:, :, 2], program[:, :, 3]
        args0 = bucketize(args0, bins).unsqueeze(dim=-1)
        args1 = bucketize(args1, bins).unsqueeze(dim=-1)
        program_discretized = torch.cat([bases, args0, args1], dim=-1)  # prog_len x bs  x 4

        return program_discretized

    def calc_acc_program(self, sampled_seq, target_seq):
        assert sampled_seq.size() == target_seq.size()

        # flatten all the labels
        sampled_primitives = sampled_seq.contiguous().view(-1, sampled_seq.size(-1))
        target_primitives = target_seq.contiguous().view(-1, target_seq.size(-1))

        # create a mask by filtering out all tokens that ARE NOT the padding token
        pad_token = torch.tensor([self.base_decoder_names.index('<PAD>'), 0, 0, 0],
                                 dtype=torch.float32).cuda()
        mask = torch.any(target_primitives != pad_token, dim=-1).to(torch.float32)  # [N*T]
        # count how many effective tokens we have
        eff_tokens = float(torch.sum(mask).item())

        acc_base_idx, acc_base_type, acc_args0, acc_args1 = \
            self.primitive_decoder.calc_acc_primitive(sampled_primitives, target_primitives)

        acc_base_idx_ave = torch.sum(acc_base_idx * mask) / eff_tokens  # a scalar tensor
        acc_base_type_ave = torch.sum(acc_base_type * mask) / eff_tokens  # a scalar tensor
        acc_args0_ave = torch.sum(acc_args0 * mask) / eff_tokens  # a scalar tensor
        acc_args1_ave = torch.sum(acc_args1 * mask) / eff_tokens  # a scalar tensor

        return acc_base_idx_ave, acc_base_type_ave, acc_args0_ave, acc_args1_ave


class PrimitiveDecoder(nn.Module):
    def __init__(self, hidden_dim=64, continuous=True, use_mixture_density=True, components_size=5,
                 base_decoder_names=BASE_ACTIONS, base_decoder_types=BASE_LINE_TYPES,
                 discretized_num=10, base_type_coef=1.0, arg_coef=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_decoder_names = base_decoder_names
        self.base_decoder_types = base_decoder_types
        self.continuous = continuous
        self.components_size = components_size
        self.use_mixture_density = use_mixture_density
        self.discretized_num = discretized_num
        self.base_type_coef = base_type_coef
        self.arg_coef = arg_coef

        if self.continuous:
            self.output_dims = [(int, len(self.base_decoder_names)),
                                (int, len(self.base_decoder_types)),
                                (float, 1),
                                (float, 1)]  # distance (0, 1), direction (0, 1)
        else:
            self.output_dims = [(int, len(self.base_decoder_names)),
                                (int, len(self.base_decoder_types)),
                                (int, self.discretized_num),
                                (int, self.discretized_num)]  # distance, directions (discretized)

        self.base_actions_size = self.output_dims[0][1]
        self.base_act_types_size = self.output_dims[1][1]
        self.args0_size = self.output_dims[2][1]
        self.args1_size = self.output_dims[3][1]

        in_dim = self.hidden_dim
        self.hidden2base_actions = nn.Linear(in_dim, self.base_actions_size)
        in_dim += self.base_actions_size
        self.hidden2base_act_types = nn.Linear(in_dim, self.base_act_types_size)
        in_dim += self.base_act_types_size
        self.hidden2args0 = MixtureDensityLayer(in_dim, self.components_size) \
            if self.continuous and self.use_mixture_density else nn.Linear(in_dim, self.args0_size)
        in_dim += self.args0_size
        self.hidden2args1 = MixtureDensityLayer(in_dim, self.components_size) \
            if self.continuous and self.use_mixture_density else nn.Linear(in_dim, self.args1_size)

    def forward(self, lstm_out, target_primitive):
        """
        target_primitive in the form of (base_func_idx, base_func_type, dist, angle)
        """
        assert target_primitive.size(1) == 4, target_primitive.size()
        target_base_idx, target_base_type, target_args0, target_args1 = \
            target_primitive[:, 0], target_primitive[:, 1], \
            target_primitive[:, 2], target_primitive[:, 3]

        # predict the base action index
        in_tensors = [lstm_out]
        pred_base_idx_logits = self.hidden2base_actions(torch.cat(in_tensors, dim=-1))
        loss_base_idx = F.cross_entropy(input=pred_base_idx_logits,
                                        target=target_base_idx.to(torch.int64),
                                        reduction='none')  # a tensor [N]
        acc_base_idx = (torch.argmax(
            pred_base_idx_logits, dim=1) == target_base_idx.to(torch.int64)
                        ).to(torch.float32)  # a tensor [N]

        # predict the base action type
        target_base_idx_onehot = F.one_hot(target_base_idx.to(torch.int64),
                                           self.base_actions_size).to(torch.float32)
        in_tensors.append(target_base_idx_onehot)
        pred_base_type_logits = self.hidden2base_act_types(torch.cat(in_tensors, dim=-1))
        loss_base_type = F.cross_entropy(input=pred_base_type_logits,
                                         target=target_base_type.to(torch.int64),
                                         reduction='none')  # a tensor [N]
        acc_base_type = (torch.argmax(
            pred_base_type_logits, dim=1) == target_base_type.to(torch.int64)
                         ).to(torch.float32)  # a tensor [N]

        # predict the args
        target_base_type_onehot = F.one_hot(target_base_type.to(torch.int64),
                                            self.base_act_types_size).to(torch.float32)
        in_tensors.append(target_base_type_onehot)
        loss_args0, loss_args1, acc_args0, acc_args1 = self.decode_args(
            in_tensors, target_args0, target_args1)

        loss_primitive = loss_base_idx + \
                         self.base_type_coef * loss_base_type + \
                         self.arg_coef * (loss_args0 + loss_args1)  # [N]

        return loss_primitive, acc_base_idx, acc_base_type, acc_args0, acc_args1

    def sample(self, lstm_out, cur_pos):
        """
        hidden is the lstm output at time i
        """
        assert lstm_out.ndim == 2, lstm_out.ndim

        # predict the base action index
        in_tensors = [lstm_out]
        pred_base_idx_logits = self.hidden2base_actions(torch.cat(in_tensors, dim=-1))

        # add one constraint: base actions only from ['line', 'arc'] for the first two actions
        if cur_pos <= 1:
            offset = 2
            pred_base_idx_probs = torch.softmax(pred_base_idx_logits[:, offset:offset + 2], dim=-1)
            assert pred_base_idx_probs.size(1) == 2
            pred_base_idx = torch.multinomial(pred_base_idx_probs, num_samples=1).to(torch.float32) + offset  # [N, 1]
        else:
            pred_base_idx_probs = torch.softmax(pred_base_idx_logits, dim=-1)
            assert pred_base_idx_probs.size(1) == self.base_actions_size
            pred_base_idx = torch.multinomial(pred_base_idx_probs, num_samples=1).to(torch.float32)  # [N, 1]

        # predict the base action type
        target_base_idx_onehot = F.one_hot(pred_base_idx.squeeze().to(torch.int64),
                                           self.base_actions_size).to(torch.float32)
        in_tensors.append(target_base_idx_onehot)
        pred_base_type_logits = self.hidden2base_act_types(torch.cat(in_tensors, dim=-1))
        pred_base_type_probs = torch.softmax(pred_base_type_logits, dim=-1)
        pred_base_type = torch.multinomial(pred_base_type_probs, num_samples=1).to(torch.float32)  # [N, 1]

        # predict the args0
        target_base_type_onehot = F.one_hot(pred_base_type.squeeze().to(torch.int64),
                                            self.base_act_types_size).to(torch.float32)
        in_tensors.append(target_base_type_onehot)
        pred_args0 = self.hidden2args0(torch.cat(in_tensors, dim=-1))
        pred_args0 = self.sample_arg(pred_args0)  # [N, 1], interval id
        if not self.continuous:
            pred_args0 = pred_args0 / self.discretized_num

        # predict the args1
        self.update_in_tensors_with_args0(in_tensors, pred_args0)
        assert len(in_tensors) == 4
        pred_args1 = self.hidden2args1(torch.cat(in_tensors, dim=-1))
        pred_args1 = self.sample_arg(pred_args1)  # [N, 1], interval id
        if not self.continuous:
            pred_args1 = pred_args1 / self.discretized_num

        sample_primitive = torch.cat([pred_base_idx, pred_base_type, pred_args0, pred_args1], dim=-1)  # [N, 4]

        return sample_primitive

    def decode_args(self, in_tensors, target_args0, target_args1):

        pred_args0 = self.hidden2args0(torch.cat(in_tensors, dim=-1))
        self.update_in_tensors_with_args0(in_tensors, target_args0)
        assert len(in_tensors) == 4, len(in_tensors)
        pred_args1 = self.hidden2args1(torch.cat(in_tensors, dim=-1))

        if self.continuous:
            if self.use_mixture_density:
                # negative log-likelihood
                loss_args0 = -self.hidden2args0.get_MD_LogLikelihood(pred_args0, target_args0)  # a tensor [N]
                loss_args1 = -self.hidden2args1.get_MD_LogLikelihood(pred_args1, target_args1)  # a tensor [N]

                # use negative MSE as "acc" metric
                pred_args0 = self.hidden2args1.sample_MD(pred_args0)  # [N, 1]
                pred_args1 = self.hidden2args1.sample_MD(pred_args1)  # [N, 1]
                acc_args0 = -(pred_args0.squeeze() - target_args0.squeeze()) ** 2  # a tensor [N]
                acc_args1 = -(pred_args1.squeeze() - target_args1.squeeze()) ** 2  # a tensor [N]

            else:
                # MSE
                loss_args0 = (pred_args0.squeeze() - target_args0.squeeze()) ** 2  # a tensor [N]
                loss_args1 = (pred_args1.squeeze() - target_args1.squeeze()) ** 2  # a tensor [N]

                # use negative MSE as "acc" metric
                acc_args0 = -loss_args0  # a tensor [N]
                acc_args1 = -loss_args1  # a tensor [N]

        else:
            # loss measuring the cross-entropy of `angle` arg (`dist` arg is a constant)
            # print('pred_args0.size(): ', pred_args0.size())
            # print('target_args0.size(): ', target_args0.size())
            loss_args0 = F.cross_entropy(input=pred_args0,
                                         target=target_args0.to(torch.int64),
                                         reduction='none')  # a tensor [N]
            acc_args0 = (torch.argmax(
                pred_args0, dim=1) == target_args0.to(torch.int64)
                         ).to(torch.float32)  # a tensor [N]

            loss_args1 = F.cross_entropy(input=pred_args1,
                                         target=target_args1.to(torch.int64),
                                         reduction='none')  # a tensor [N]
            acc_args1 = (torch.argmax(
                pred_args1, dim=1) == target_args1.to(torch.int64)
                         ).to(torch.float32)  # a tensor [N]

        return loss_args0, loss_args1, acc_args0, acc_args1

    def sample_arg(self, pred_arg):
        if not self.continuous:
            # pred_arg is logits [N, L]
            pred_arg_probs = torch.softmax(pred_arg, dim=-1)
            pred_arg = torch.multinomial(pred_arg_probs, num_samples=1).to(torch.float32)  # [N, 1]
        elif self.use_mixture_density:
            # pred_arg is actually (u, v, p)
            pred_arg = self.hidden2args1.sample_MD(pred_arg)  # [N, 1]
        else:
            pred_arg = pred_arg.clamp(min=0.01, max=1)  # [N, 1]
        return pred_arg

    def update_in_tensors_with_args0(self, in_tensors, targe_or_pred_args0):
        if self.continuous:
            if targe_or_pred_args0.ndim == 1:
                targe_or_pred_args0.unsqueeze_(dim=-1)  # [N, 1]
            in_tensors.append(targe_or_pred_args0)
        else:
            args0_onehot = F.one_hot(targe_or_pred_args0.squeeze().to(torch.int64),
                                     self.args0_size).to(torch.float32)
            in_tensors.append(args0_onehot)

    def calc_acc_primitive(self, sampled_primitive, target_primitive):
        assert sampled_primitive.size() == target_primitive.size()
        sampled_base_idx, sampled_base_type, sampled_args0, sampled_args1 = \
            sampled_primitive[:, 0], sampled_primitive[:, 1], sampled_primitive[:, 2], sampled_primitive[:, 3]
        target_base_idx, target_base_type, target_args0, target_args1 = \
            target_primitive[:, 0], target_primitive[:, 1], target_primitive[:, 2], target_primitive[:, 3]

        acc_base_idx = (sampled_base_idx.to(torch.int64) == target_base_idx.to(torch.int64)).to(
            torch.float32)  # a tensor [N]
        acc_base_type = (sampled_base_type.to(torch.int64) == target_base_type.to(torch.int64)).to(
            torch.float32)  # a tensor [N]

        if self.continuous:
            # MSE
            acc_args0 = -(sampled_args0.squeeze() - target_args0.squeeze()) ** 2  # a tensor [N]
            acc_args1 = -(sampled_args1.squeeze() - target_args1.squeeze()) ** 2  # a tensor [N]

        else:
            acc_args0 = (sampled_args0.to(torch.int64) == target_args0.to(torch.int64)).to(
                torch.float32)  # a tensor [N]
            acc_args1 = (sampled_args1.to(torch.int64) == target_args1.to(torch.int64)).to(
                torch.float32)  # a tensor [N]

        return acc_base_idx, acc_base_type, acc_args0, acc_args1


# ###################### MixtureDensity ########################
class MixtureDensityLayer(nn.Module):
    def __init__(self, input_size, components_size, epsilon=1e-4, bounds=(0, 1)):
        super().__init__()
        self.components_size = components_size
        self.input_size = input_size
        self.epsilon = epsilon
        self.bounds = bounds

        self.u_layer = nn.Linear(self.input_size, self.components_size)
        self.v_layer = nn.Linear(self.input_size, self.components_size)
        self.p_layer = nn.Linear(self.input_size, self.components_size)

    def forward(self, inputs):
        u = self.u_layer(inputs)
        if self.bounds is not None:
            (upper, lower) = self.bounds
            d = upper - lower
            u = d * torch.sigmoid(u) + lower
        v = F.softplus(self.v_layer(inputs)) + self.epsilon
        p = F.log_softmax(self.p_layer(inputs), dim=-1)
        return [u, v, p]

    def get_MD_LogLikelihood(self, MD_params, target):
        if target.ndim == 1:
            target = target.unsqueeze(dim=-1)
        u, v, p = MD_params[0], MD_params[1], MD_params[2]
        d = u - torch.cat([target] * self.components_size, dim=-1)
        logLikelihoods = -d * d * torch.reciprocal(2.0 * v) - 0.5 * torch.log(v) + p

        # normalizing constant
        logLikelihoods -= 0.39908993417  # -log(1/sqrt(2pi))

        return torch.logsumexp(logLikelihoods, dim=-1)  # size: [N]

    def sample_MD(self, MD_params):
        u, v, p = MD_params[0], MD_params[1], MD_params[2]
        j = torch.multinomial(torch.exp(p), num_samples=1)  # size: [N, 1]
        u_j = u.gather(dim=1, index=j)
        v_j = v.gather(dim=1, index=j)
        sample = torch.randn_like(u_j) * (v_j ** 0.5) + u_j  # size: [N, 1]
        return sample.clamp(min=0.01, max=1)
