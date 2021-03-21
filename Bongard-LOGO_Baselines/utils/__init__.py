# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from typing import Any
import sys
from . import few_shot

import glob
import PIL.Image
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.shape_program import prog_idx2prog_str
from bongard import bongard_painter, bongard

_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                       or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(params, name, lr, weight_decay=None, milestones=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    if milestones:
        lr_scheduler = MultiStepLR(optimizer, milestones)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def visualize_dataset(dataset, name, writer, n_samples=1):
    def get_data(dataset, i):
        if dataset.use_moco:
            return dataset.convert_raw(dataset[i][0][0])
        else:
            return dataset.convert_raw(dataset[i][0])

    for task_id in np.random.choice(dataset.n_tasks, n_samples, replace=False):
        pos_indices = [task_id * dataset.bong_size * 2 + i
                       for i in range(dataset.bong_size)]
        data_per_task_pos = torch.stack([get_data(dataset, i)
                                         for i in pos_indices])
        neg_indices = [task_id * dataset.bong_size * 2 + i + dataset.bong_size
                       for i in range(dataset.bong_size)]
        data_per_task_neg = torch.stack([get_data(dataset, i)
                                         for i in neg_indices])

        if name is None:
            name = os.path.basename(dataset.tasks[task_id])
        else:
            name += '_' + os.path.basename(dataset.tasks[task_id])
        # stack 'L' to 'RGB' for visualization
        data_per_task_pos = torch.cat(
            [data_per_task_pos, data_per_task_pos, data_per_task_pos], dim=1)
        data_per_task_neg = torch.cat(
            [data_per_task_neg, data_per_task_neg, data_per_task_neg], dim=1)
        writer.add_images('visualize_' + name + '_task' + str(task_id) + '/pos',
                          data_per_task_pos)
        writer.add_images('visualize_' + name + '_task' + str(task_id) + '/neg',
                          data_per_task_neg)
    writer.flush()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0:  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------

def save_images2(images_A, images_B, size, image_path, is_permute):
    img = merge2(images_A, images_B, size, is_permute=is_permute)
    assert img.ndim == 3
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    fmt = 'RGB' if img.ndim == 3 else 'L'
    PIL.Image.fromarray(img, fmt).save(image_path)


def merge2(images_A, images_B, size=(3, 2), gap=20, is_permute=False):
    assert images_A.shape[0] == images_B.shape[0]
    h, w, c = images_A.shape[1], images_A.shape[2], images_A.shape[-1]

    test_idx_vec = np.array([0, 1])
    if is_permute:
        test_idx_vec = np.random.permutation([0, 1])

    test_images = [images_A[-1], images_B[-1]]

    img = np.zeros((h * size[0] + 2 * 2, w * size[1] * 2 + 3 * gap + 2 * 4 + w, c)) + 200
    for idx, image in enumerate(images_A[:-1]):
        i = idx % size[1]
        j = idx // size[1]
        pre_i = 0 if i == 0 else 2
        pre_j = 0 if j == 0 else 2
        img[h * j + pre_j * j:h * (j + 1) + pre_j * j, w * i + pre_i * i:w * (i + 1) + pre_i * i, :] = image

    for idx, image in enumerate(images_B[:-1]):
        i = idx % size[1] + size[1]
        j = idx // size[1]
        pre_i = 0 if i == 0 else 2
        pre_j = 0 if j == 0 else 2
        img[h * j + pre_j * j:h * (j + 1) + pre_j * j, gap + w * i + pre_i * i:gap + w * (i + 1) + pre_i * i, :] = image

    img[:h, 3 * gap + w * 4 + 2 * 4: 3 * gap + w * 5 + 2 * 4, :] = test_images[test_idx_vec[0]]
    img[h + 2: h * 2 + 2, 3 * gap + w * 4 + 2 * 4: 3 * gap + w * 5 + 2 * 4, :] = test_images[test_idx_vec[1]]

    return img


def vis_bongard(data_path, vis_path):
    filenames_A = sorted(glob.glob(os.path.join(data_path, '1', '*.png')))
    images_A_all = np.array([np.array(PIL.Image.open(fname)) for fname in filenames_A])
    print('images_A_all.shape: ', images_A_all.shape)
    filenames_B = sorted(glob.glob(os.path.join(data_path, '0', '*.png')))
    print('filenames_A[:1]: {}, filenames_B[:1]: {}'.format(filenames_A[:1], filenames_B[:1]))
    images_B_all = np.array([np.array(PIL.Image.open(fname)) for fname in filenames_B])

    save_images2(images_A_all, images_B_all, (3, 2), '{}/bongard_infer.png'.format(vis_path), is_permute=False)


def visualize_reconstructions(dataset, sample_func, name=None, n_samples=1, save_path='reconst_samples'):
    def get_data(dataset, i):
        return dataset.convert_raw(dataset[i][0]), dataset[i][1]

    for task_id in np.random.choice(dataset.n_tasks, n_samples, replace=False):
        if name is None:
            name = os.path.basename(dataset.tasks[task_id])
        else:
            name += '_' + os.path.basename(dataset.tasks[task_id])

        # gt positive data and programs
        pos_indices = [task_id * dataset.bong_size * 2 + i
                       for i in range(dataset.bong_size)]

        data_per_task_pos = torch.stack([get_data(dataset, i)[0] for i in pos_indices])
        data_per_task_pos_np = np.array([np.array(transforms.ToPILImage()(get_data(dataset, i)[0]).convert('RGB'))
                                         for i in pos_indices])
        print('data_per_task_pos_np size: ', data_per_task_pos_np.shape)
        prog_per_task_pos_lst = [get_data(dataset, i)[1].tolist() for i in pos_indices]
        prog_per_task_pos = torch.stack([get_data(dataset, i)[1] for i in pos_indices])
        prog_str_per_task_pos = [prog_idx2prog_str(prog) for prog in prog_per_task_pos_lst]
        # print('prog_str_per_task_pos: ', prog_str_per_task_pos)

        # gt negative data and programs
        neg_indices = [task_id * dataset.bong_size * 2 + i + dataset.bong_size
                       for i in range(dataset.bong_size)]

        data_per_task_neg = torch.stack([get_data(dataset, i)[0] for i in neg_indices])
        data_per_task_neg_np = np.array([np.array(transforms.ToPILImage()(get_data(dataset, i)[0]).convert('RGB'))
                                         for i in neg_indices])
        prog_per_task_neg_lst = [get_data(dataset, i)[1].tolist() for i in neg_indices]
        prog_per_task_neg = torch.stack([get_data(dataset, i)[1] for i in neg_indices])
        prog_str_per_task_neg = [prog_idx2prog_str(prog) for prog in prog_per_task_neg_lst]
        # print('prog_str_per_task_neg: ', prog_str_per_task_neg)

        # store the ground-truth images
        save_dir = os.path.join(save_path, name)
        os.makedirs(save_dir, exist_ok=True)
        save_images2(data_per_task_pos_np, data_per_task_neg_np, size=(3, 2),
                     image_path=os.path.join(save_dir, 'bongard_gt.png'), is_permute=False)

        # store the ground-truth programs
        with open(os.path.join(save_dir, 'programs_gt_inferred.txt'), 'w') as f:
            f.write('prog_str_per_task_pos (gt): {} \n'.format(prog_str_per_task_pos))
            f.write('prog_str_per_task_neg (gt): {} \n'.format(prog_str_per_task_neg))

        # infer programs from gt images and transform to programs string
        programs_pos_inferred, _, acc_base_idx_pos, acc_base_type_pos, acc_args0_pos, acc_args1_pos = \
            sample_func(data_per_task_pos.cuda(), prog_per_task_pos.cuda())
        programs_neg_inferred, _, acc_base_idx_neg, acc_base_type_neg, acc_args0_neg, acc_args1_neg = \
            sample_func(data_per_task_neg.cuda(), prog_per_task_neg.cuda())

        log('inference-time acc (pos and neg) in {}: '
            '({:.4f}, {:.4f}, {:.4f}, {:.4f}) and ({:.4f}, {:.4f}, {:.4f}, {:.4f})'
            .format(name, acc_base_idx_pos, acc_base_type_pos, acc_args0_pos, acc_args1_pos,
                    acc_base_idx_neg, acc_base_type_neg, acc_args0_neg, acc_args1_neg))

        programs_str_pos_inferred = [prog_idx2prog_str(prog) for prog in programs_pos_inferred.tolist()]
        # print('programs_pos_inferred:', programs_pos_inferred)
        # print('programs_str_pos_inferred:', programs_str_pos_inferred)
        programs_str_neg_inferred = [prog_idx2prog_str(prog) for prog in programs_neg_inferred.tolist()]
        # print('programs_neg_inferred:', programs_neg_inferred)
        # print('programs_str_neg_inferred:', programs_str_neg_inferred)

        # store the inferred programs
        with open(os.path.join(save_dir, 'programs_gt_inferred.txt'), 'a+') as f:
            f.write('prog_str_per_task_pos (inferred): {} \n'.format(programs_str_pos_inferred))
            f.write('prog_str_per_task_neg (inferred): {} \n'.format(programs_str_neg_inferred))

        # decide if inferred programs are valid to draw images
        is_valid_inferred = True
        for i in range(dataset.bong_size):
            if len(programs_str_pos_inferred[i]) > 2 or len(programs_str_neg_inferred[i]) > 2:
                print('[Warning] Inferred programs is invalid due to [{}] shapes in an image'.format(
                    max(len(programs_str_pos_inferred[i]), len(programs_str_neg_inferred[i]))
                ))
                is_valid_inferred = False
                break
        if not is_valid_inferred:
            continue

        # save reconstructed problems (programs_str => saved images)
        bongard_problem_positive_images = []
        for prog_per_img in programs_str_pos_inferred:
            bongard_pos_image_shapes = []
            for prog_per_shape in prog_per_img:
                base_actions = [bongard.ArcAction.import_from_action_string(base_action)
                                if base_action.split('_')[0] == 'arc' else
                                bongard.LineAction.import_from_action_string(base_action)
                                for base_action in prog_per_shape]
                shape = bongard.OneStrokeShape(basic_actions=base_actions,
                                               start_coordinates=None, start_orientation=None)
                bongard_pos_image_shapes.append(shape)
            bongard_image = bongard.BongardImage(one_stroke_shapes=bongard_pos_image_shapes)
            bongard_problem_positive_images.append(bongard_image)
        bongard_problem_negative_images = []
        for prog_per_img in programs_str_neg_inferred:
            bongard_neg_image_shapes = []
            for prog_per_shape in prog_per_img:
                base_actions = [bongard.ArcAction.import_from_action_string(base_action)
                                if base_action.split('_')[0] == 'arc' else
                                bongard.LineAction.import_from_action_string(base_action)
                                for base_action in prog_per_shape]
                shape = bongard.OneStrokeShape(basic_actions=base_actions,
                                               start_coordinates=None, start_orientation=None)
                bongard_neg_image_shapes.append(shape)
            bongard_image = bongard.BongardImage(one_stroke_shapes=bongard_neg_image_shapes)
            bongard_problem_negative_images.append(bongard_image)
        bongard_problem_name = name
        bongard_problem = bongard.BongardProblem(positive_bongard_images=bongard_problem_positive_images,
                                                 negative_bongard_images=bongard_problem_negative_images,
                                                 problem_name=bongard_problem_name, positive_rules=None,
                                                 negative_rules=None)

        # keep the scaling_factors_range to be the same with real problems
        if 'ff' in name:
            scaling_factors_range = (80, 140)
        elif 'hd' in name:
            scaling_factors_range = (120, 240)
        elif 'bd' in name:
            scaling_factors_range = (120, 240)
        else:
            scaling_factors_range = (140, 150)
        problem_painter = bongard_painter.BongardProblemPainter(scaling_factors_range=scaling_factors_range,
                                                                random_seed=0)

        problem_painter.create_bongard_problem(
            bongard_problem,
            bongard_problem_ps_dir=os.path.join(save_path, 'ps', bongard_problem_name),
            bongard_problem_png_dir=os.path.join(save_path, 'png', bongard_problem_name))

        vis_bongard(data_path=os.path.join(save_path, 'png', bongard_problem_name), vis_path=save_dir)

# ----------------------------------------------------------------------------
