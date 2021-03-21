# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .models import register


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


@register('lstm')
class LSTMModel(nn.Module):
    """Container module with a recurrent module only"""

    def __init__(self, nemb, nhid, nlayers, dropout=0.5, repackage=True, **kwargs):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(nemb, nhid, nlayers, dropout=dropout, batch_first=False)

        self.model_type = 'lstm'
        self.nhid = nhid
        self.nlayers = nlayers
        self.repackage = repackage

    def forward(self, emb, hidden, img_feat=None):

        if self.repackage:
            hidden = repackage_hidden(hidden)

        emb = self.drop(emb)  # seq_len x bs x nemb
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)  # seq_len x bs x nhid
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

    def flatten_parameters(self):
        self.lstm.flatten_parameters()


@register('transformer')
class TransformerModel(nn.Module):

    def __init__(self, nemb, nhead, nhid, nlayers, dropout=0.5, **kwargs):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(nemb, dropout)
        encoder_layers = TransformerEncoderLayer(nemb, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.nemb = nemb

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src_emb, has_mask=True):
        if has_mask:
            device = src_emb.device
            if self.src_mask is None or self.src_mask.size(0) != len(src_emb):
                mask = self._generate_square_subsequent_mask(len(src_emb)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src_emb = src_emb * math.sqrt(self.nemb)
        src_emb = self.pos_encoder(src_emb)  # seq_len x bs x nemb
        output = self.transformer_encoder(src_emb, self.src_mask)  # seq_len x bs x nhid

        # also provide a hidden (embedding) for use, here we apply averaging over seq for simplicity
        hidden = torch.mean(output, dim=0)  # bs x nhid

        return output, hidden


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
