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

from glom_src import *


# @register('meta-baseline')
# class MetaBaseline(nn.Module):

#     def __init__(self, encoder, encoder_args={}, method='cos',
#                  temp=10., temp_learnable=True):
#         super().__init__()
#         self.encoder = models.make(encoder, **encoder_args)
#         self.method = method

#         if temp_learnable:
#             self.temp = nn.Parameter(torch.tensor(temp))
#         else:
#             self.temp = temp

#     def forward(self, x_shot, x_query, **kwargs):
#         shot_shape = x_shot.shape[:-3]
#         query_shape = x_query.shape[:-3]
#         img_shape = x_shot.shape[-3:]

#         x_shot = x_shot.view(-1, *img_shape)
#         x_query = x_query.view(-1, *img_shape)
#         x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
#         x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
#         x_shot = x_shot.view(*shot_shape, -1)
#         x_query = x_query.view(*query_shape, -1)

#         if self.method == 'cos':
#             x_shot = x_shot.mean(dim=-2)
#             x_shot = F.normalize(x_shot, dim=-1)  # [ep_per_batch, way, feature_len]
#             x_query = F.normalize(x_query, dim=-1)  # [ep_per_batch, way * query, feature_len]
#             metric = 'dot'
#         elif self.method == 'sqr':
#             x_shot = x_shot.mean(dim=-2)
#             metric = 'sqr'

#         logits = utils.compute_logits(
#                 x_query, x_shot, metric=metric, temp=self.temp)  # [ep_per_batch, way * query, way]
#         return logits

@register('meta-baseline')
class GLOM_ProtoNet(nn.Module):
  def __init__(self, encoder, encoder_args= {}):
    super(GLOM_ProtoNet, self).__init__()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # self.encoder = GLOM_encoder(in_channels=1, num_timesteps=10, image_dim=(64,64), patch_dim=3, number_of_embeddings=5, vector_dim=128, device=device).cuda()
    # self.encoder.load_state_dict(torch.load('/home/t-shahhet/glom/runs/bongard_0/encoder.pt'))
    # self.num_patches = self.encoder.get_num_patches()
    self.encoder = GLOM_CNN([2, 4, 6, 8, 10], alpha=0.33, beta=0.33, gamma=0.33, img_dim=(1, 512, 512), num_timesteps=10)
    self.vector_dim = 128

    self.temp = nn.Parameter(torch.tensor(10.))

  def forward(self, x_shot, x_query, **kwargs):
    shot_shape = x_shot.shape[:-3]
    query_shape = x_query.shape[:-3]
    img_shape = x_shot.shape[-3:]

    x_shot = x_shot.view(-1, *img_shape)
    x_query = x_query.view(-1, *img_shape)

    x_tot, loss = self.encoder(torch.cat([x_shot, x_query], dim=0), mode='full')
    x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]

    x_shot = x_shot.reshape((shot_shape[0], shot_shape[1], shot_shape[2], x_shot.shape[-1]))
    x_query = x_query.reshape((query_shape[0], query_shape[1], x_query.shape[-1]))

    x_shot = x_shot.mean(dim=-2)
    x_shot = F.normalize(x_shot, dim=-1)  # [ep_per_batch, way, feature_len]
    x_query = F.normalize(x_query, dim=-1)  # [ep_per_batch, way * query, feature_len]
    metric = 'dot'

    logits = utils.compute_logits(x_query, x_shot, metric=metric, temp=self.temp) 

    return logits, loss