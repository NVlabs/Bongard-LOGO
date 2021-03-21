# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import os
import json
from PIL import Image
import numpy as np
import glob
from PIL import ImageFilter
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register


@register('shape-bongard-ff')
class ShapeBongard_FF(Dataset):

    def __init__(self, root_path, image_size=512, box_size=512, **kwargs):
        self.bong_size = 7
        if box_size is None:
            box_size = image_size

        self.tasks = sorted(os.listdir(os.path.join(root_path, 'images')))
        if kwargs.get('split'):
            path = kwargs.get('split_file')
            if path is None:
                path = os.path.join(root_path.rstrip('/'), 'ShapeBongard_FF_split.json')
            split = json.load(open(path, 'r'))
            self.tasks = sorted(split[kwargs['split']])
        self.n_tasks = len(self.tasks)

        task_paths = [os.path.join(root_path, 'images', task) for task in self.tasks]
        self.file_paths = []
        self.labels = []
        for task_path in task_paths:
            self.file_paths.extend(sorted(glob.glob(os.path.join(task_path, '1', '*.png'))))
            self.labels.extend([1 for _ in range(self.bong_size)])
            self.file_paths.extend(sorted(glob.glob(os.path.join(task_path, '0', '*.png'))))
            self.labels.extend([0 for _ in range(self.bong_size)])
        assert len(self.file_paths) == self.bong_size * 2 * len(task_paths)
        assert len(self.labels) == len(self.file_paths)

        norm_params = {'mean': [0.5], 'std': [0.5]}  # grey-scale to [-1, 1]
        normalize = transforms.Normalize(**norm_params)

        self.use_moco = False
        if kwargs.get('moco'):
            self.use_moco = kwargs['moco']

        if self.use_moco:
            if kwargs.get('aug_plus'):
                # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
                self.transform = TwoCropsTransform(transforms.Compose([
                    # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.Resize(image_size),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]))
            else:
                # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
                self.transform = TwoCropsTransform(transforms.Compose([
                    # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                    transforms.Resize(image_size),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]))

        else:
            if kwargs.get('augment'):
                self.transform = transforms.Compose([
                    # transforms.RandomResizedCrop(image_size),
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(box_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize,
                ])

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(-1).type_as(x)
            std = torch.tensor(norm_params['std']).view(-1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, i):
        img = Image.open(self.file_paths[i]).convert('L')
        return self.transform(img), self.labels[i]


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
