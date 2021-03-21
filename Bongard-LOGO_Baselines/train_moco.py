# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import BongardSampler


def main(config):
    svname = args.name
    if svname is None:
        svname = 'moco_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
        out_dim = config['model_args']['encoder_args']['out_dim']
        svname += '-out_dim' + str(out_dim)
    svname += '-seed' + str(args.seed)
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join(args.save_dir, svname)
    utils.ensure_path(save_path, remove=False)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    random_state = np.random.RandomState(args.seed)
    print('seed：', args.seed)

    logger = utils.Logger(file_name=os.path.join(save_path, "log_sdout.txt"), file_mode="a+", should_flush=True)

    #### Dataset ####

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    utils.log('train dataset: {} (x{})'.format(
        train_dataset[0][0][0].shape, len(train_dataset)))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    # val
    if config.get('val_dataset'):
        eval_val = True
        val_dataset = datasets.make(config['val_dataset'],
                                    **config['val_dataset_args'])
        val_loader = DataLoader(val_dataset, config['batch_size'],
                                num_workers=8, pin_memory=True, drop_last=True)
        utils.log('val dataset: {} (x{})'.format(
            val_dataset[0][0][0].shape, len(val_dataset)))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    else:
        eval_val = False

    # few-shot eval
    if config.get('eval_fs'):
        ef_epoch = config.get('eval_fs_epoch')
        if ef_epoch is None:
            ef_epoch = 5
        eval_fs = True
        n_way = 2
        n_query = 1
        n_shot = 6

        if config.get('ep_per_batch') is not None:
            ep_per_batch = config['ep_per_batch']
        else:
            ep_per_batch = 1

        # tvals
        fs_loaders = {}
        tval_name_ntasks_dict = {'tval': 2000, 'tval_ff': 600, 'tval_bd': 480,
                                 'tval_hd_comb': 400, 'tval_hd_novel': 320}  # numbers depend on dataset
        for tval_type in tval_name_ntasks_dict.keys():
            if config.get('{}_dataset'.format(tval_type)):
                tval_dataset = datasets.make(config['{}_dataset'.format(tval_type)],
                                             **config['{}_dataset_args'.format(tval_type)])
                utils.log('{} dataset: {} (x{})'.format(
                    tval_type, tval_dataset[0][0][0].shape, len(tval_dataset)))
                if config.get('visualize_datasets'):
                    utils.visualize_dataset(tval_dataset, 'tval_ff_dataset', writer)
                tval_sampler = BongardSampler(
                    tval_dataset.n_tasks, n_batch=tval_name_ntasks_dict[tval_type] // ep_per_batch,
                    ep_per_batch=ep_per_batch, seed=random_state.randint(2 ** 31))
                tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                         num_workers=8, pin_memory=True)
                fs_loaders.update({tval_type: tval_loader})
            else:
                fs_loaders.update({tval_type: None})

    else:
        eval_fs = False

    ########

    #### Model and Optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

    if eval_fs:
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if eval_fs:
            fs_model = nn.DataParallel(fs_model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], **config['optimizer_args'])

    ########

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    for epoch in range(1, max_epoch + 1 + 1):

        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va', 'tvl', 'tva']
        if eval_fs:
            for k, v in fs_loaders.items():
                if v is not None:
                    aves_keys += ['fsa' + k.split('tval')[-1]]
        aves = {ave_k: utils.Averager() for ave_k in aves_keys}

        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, _ in tqdm(train_loader, desc='train', leave=False):
            logits, label = model(im_q=data[0].cuda(), im_k=data[1].cuda())

            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None;
            loss = None

        # val
        if eval_val:
            model.eval()
            for data, _ in tqdm(val_loader, desc='val', leave=False):
                with torch.no_grad():
                    logits, label = model(im_q=data[0].cuda(), im_k=data[1].cuda())
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)

                aves['vl'].add(loss.item())
                aves['va'].add(acc)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            fs_model.eval()
            for k, v in fs_loaders.items():
                if v is not None:
                    ave_key = 'fsa' + k.split('tval')[-1]
                    np.random.seed(0)
                    for data, _ in tqdm(v, desc=ave_key, leave=False):
                        x_shot, x_query = fs.split_shot_query(
                            data[0].cuda(), n_way, n_shot, n_query, ep_per_batch=ep_per_batch)
                        label_query = fs.make_nk_label(
                            n_way, n_query, ep_per_batch=ep_per_batch).cuda()
                        with torch.no_grad():
                            logits = fs_model(x_shot, x_query).view(-1, n_way)
                            acc = utils.compute_acc(logits, label_query)
                        aves[ave_key].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        if epoch <= max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
            epoch_str, aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)

        if eval_val:
            log_str += ', val {:.4f}|{:.4f}, tval {:.4f}|{:.4f}'.format(
                aves['vl'], aves['va'], aves['tvl'], aves['tva'])
            writer.add_scalars('loss', {'val': aves['vl']}, epoch)
            writer.add_scalars('loss', {'tval': aves['tvl']}, epoch)
            writer.add_scalars('acc', {'val': aves['va']}, epoch)
            writer.add_scalars('acc', {'tval': aves['tva']}, epoch)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            log_str += ', fs'
            for ave_key in aves_keys:
                if 'fsa' in ave_key:
                    log_str += ' {}: {:.4f}'.format(ave_key, aves[ave_key])
                    writer.add_scalars('acc', {ave_key: aves[ave_key]}, epoch)

        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        if epoch <= max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj, os.path.join(
                    save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))

        writer.flush()

    print('finished training!')
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--save_dir', default='./save')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
