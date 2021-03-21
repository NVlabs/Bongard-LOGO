# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import re

import models
import utils
from .models import register
from utils.few_shot import make_nk_label


@register('maml')
class MAML(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='anil', step_size=0.4,
                 first_order=False):
        super().__init__()
        self.n_way = 2
        self.n_shot = 6
        self.step_size = step_size
        self.first_order = first_order

        self.model = MAML_ResNet(encoder, encoder_args, n_way=self.n_way, method=method)

        print('MAML method: ', method)

    def forward(self, x_shot, x_query, **kwargs):
        ep_per_batch, n_way, n_shot = x_shot.shape[:-3]
        assert n_shot == self.n_shot and n_way == self.n_way
        img_shape = x_shot.shape[-3:]

        if kwargs.get('eval') is None:
            kwargs['eval'] = False

        first_order = self.first_order or kwargs['eval']

        x_shot = x_shot.view(-1, *img_shape)
        train_logit = self.model(x_shot)  # [bs * n_way * n_shot, n_way]
        labels_support = make_nk_label(n_way, n_shot, ep_per_batch)  # [bs * n_way * n_shot]

        inner_loss = F.cross_entropy(train_logit, labels_support.cuda())

        # self.zero_grad()
        params = gradient_update_parameters(self.model, inner_loss, step_size=self.step_size,
                                            first_order=first_order)

        x_query = x_query.view(-1, *img_shape)
        test_logit = self.model(x_query, params=params)  # [bs * n_way * n_query, n_way]

        return test_logit

    def set_first_order(self, first_order):
        self.first_order = first_order


def gradient_update_parameters(model,
                               loss,
                               params=None,
                               step_size=0.5,
                               first_order=False):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.
    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.
    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.
    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.
    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    Returns
    -------
    updated_params : `collections.OrderedDict` instance
        Dictionary containing the updated meta-parameters of the model, with one
        gradient update wrt. the inner-loss.
    """
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

    return updated_params


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.
    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """
    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param


class MetaLinear(nn.Linear, MetaModule):
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.linear(input, params['weight'], bias)


def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None

    if (key is None) or (key == ''):
        return dictionary

    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    # Compatibility with DataParallel
    if not any(filter(key_re.match, dictionary.keys())):
        key_re = re.compile(r'^module\.{0}\.(.+)'.format(re.escape(key)))

    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)


class MAML_ResNet(MetaModule):
    def __init__(self, encoder, encoder_args={}, n_way=2, method='anil'):
        super(MAML_ResNet, self).__init__()
        self.method = method
        self.encoder = models.make(encoder, **encoder_args)

        # Only the last (linear) layer is used for adaptation in ANIL
        self.classifier = MetaLinear(self.encoder.out_dim, n_way)

    def forward(self, inputs, params=None):
        if self.method == 'anil':
            features = self.encoder(inputs)
        elif self.method == 'maml':
            features = self.encoder(inputs, params=get_subdict(params, 'features'))
        else:
            raise Exception('method has to be [maml, anil].')
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return logits

