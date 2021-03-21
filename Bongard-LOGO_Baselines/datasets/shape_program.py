# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import os
import json
from PIL import Image
import numpy as np
import glob
import copy

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register

BASE_ACTIONS = ['<PAD>', 'start', 'line', 'arc', 'and', 'stop']
BASE_LINE_TYPES = ['normal', 'zigzag', 'circle', 'triangle', 'square']
MAX_LEN_PROGRM = 21


@register('shape-program')
class ShapeProgram(Dataset):

    def __init__(self, root_path, image_size=512, box_size=512, **kwargs):
        self.bong_size = 7
        if box_size is None:
            box_size = image_size

        self.tasks = []
        for prob_type in ['ff', 'bd', 'hd']:
            tasks_per_prob_type = sorted(os.listdir(os.path.join(root_path, prob_type, 'images')))
            self.tasks.extend(tasks_per_prob_type)
        if kwargs.get('split'):
            path = kwargs.get('split_file')
            if path is None:
                path = os.path.join(root_path.rstrip('/'), 'ShapeBongard_V2_split.json')
            split = json.load(open(path, 'r'))
            self.tasks = sorted(split[kwargs['split']])
        self.n_tasks = len(self.tasks)

        program_path = kwargs.get('program_file')
        if program_path is None:
            programs = {}
            for prob_type in ['ff', 'bd', 'hd']:
                program_path = os.path.join(root_path.rstrip('/'), prob_type,
                                            '{}_action_programs.json'.format(prob_type))
                programs.update(json.load(open(program_path, 'r')))
        else:
            programs = json.load(open(program_path, 'r'))

        task_paths = [os.path.join(root_path, task.split('_')[0], 'images', task) for task in self.tasks]
        self.file_paths = []
        self.parsed_programs = []
        for i, task_path in enumerate(task_paths):

            # get images for each task
            self.file_paths.extend(sorted(glob.glob(os.path.join(task_path, '1', '*.png'))))
            self.file_paths.extend(sorted(glob.glob(os.path.join(task_path, '0', '*.png'))))

            # get and parse programs for each task
            programs_pos_per_problem, programs_neg_per_problem = programs[self.tasks[i]]
            assert isinstance(programs_pos_per_problem, list), type(programs_pos_per_problem)
            assert isinstance(programs_neg_per_problem, list), type(programs_neg_per_problem)
            for program in programs_pos_per_problem:
                self.parsed_programs.append(prog_str2prog_idx(program))
            for program in programs_neg_per_problem:
                self.parsed_programs.append(prog_str2prog_idx(program))

        assert len(self.file_paths) == self.bong_size * len(task_paths) * 2
        assert len(self.parsed_programs) == len(self.file_paths)

        norm_params = {'mean': [0.5], 'std': [0.5]}  # grey-scale to [-1, 1]
        normalize = transforms.Normalize(**norm_params)

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
        return self.transform(img), torch.tensor(self.parsed_programs[i], dtype=torch.float32)


def prog_str2prog_idx(prog_str):
    """
    [['line_circle_0.700-0.500', ...], ['arc_normal_0.500_0.648-0.169', ...]]
    =>
    [[2, 0, 0, 0], [0, 2, 0.700, 0.500], ..., [3, 0, 0, 0], [1, 0, 0.648, 0.169], ..., [4, 0, 0, 0]]
    """
    assert isinstance(prog_str, list), type(prog_str)
    n_subprogs = len(prog_str)
    parsed_program = [[BASE_ACTIONS.index('start'), 0, 0, 0]]
    for i, sub_prog in enumerate(prog_str):
        for primitive in sub_prog:
            primitive_str_list = primitive.split('_')
            base_idx = BASE_ACTIONS.index(primitive_str_list[0])
            base_type = BASE_LINE_TYPES.index(primitive_str_list[1])
            args0, args1 = primitive_str_list[-1].split('-')
            args0, args1 = float(args0), float(args1)
            parsed_program.append([base_idx, base_type, args0, args1])
        if i < n_subprogs - 1:
            parsed_program.append([BASE_ACTIONS.index('and'), 0, 0, 0])

    parsed_program.append([BASE_ACTIONS.index('stop'), 0, 0, 0])

    # fill in the match the maximum length of the program sequence
    for _ in range(len(parsed_program), MAX_LEN_PROGRM):
        parsed_program.append([BASE_ACTIONS.index('<PAD>'), 0, 0, 0])

    assert len(parsed_program) == MAX_LEN_PROGRM
    return parsed_program


def prog_idx2prog_str(prog_idx):
    """
    [[2, 0, 0, 0], [0, 2, 0.700, 0.500], ..., [3, 0, 0, 0], [1, 0, 0.648, 0.169], ..., [4, 0, 0, 0]]
    =>
    [['line_circle_0.700-0.500', ...], ['arc_normal_0.500_0.648-0.169', ...]]
    """
    assert isinstance(prog_idx, list), type(prog_idx)
    assert BASE_ACTIONS[int(prog_idx[0][0])] == 'start'
    program_str = []
    sub_prog = []
    has_stop = False
    for primitive in prog_idx[1:]:
        base_name = BASE_ACTIONS[int(primitive[0])]
        base_type_name = BASE_LINE_TYPES[int(primitive[1])]
        args0 = "{:.3f}".format(primitive[2])
        args1 = "{:.3f}".format(primitive[3])
        args = '-'.join([args0, args1])
        if base_name == 'line':
            sub_prog.append('_'.join([base_name, base_type_name, args]))
        elif base_name == 'arc':
            args_arc = "{:.3f}".format(0.5)
            sub_prog.append('_'.join([base_name, base_type_name, args_arc, args]))
        elif base_name == 'and':
            program_str.append(copy.deepcopy(sub_prog))
            sub_prog = []
        elif base_name == 'stop':
            has_stop = True
            if len(sub_prog) > 0:
                program_str.append(copy.deepcopy(sub_prog))
            break
        else:
            print('[Skip it!] base_name: {} is not expected!'.format(base_name))

    if not has_stop:
        # use the maximum size of a token (9) if there is no 'stop' token
        max_len_token = 9
        program_str.append(copy.deepcopy(sub_prog[:max_len_token]))

    return program_str
