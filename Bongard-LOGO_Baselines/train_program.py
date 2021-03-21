# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import argparse
import os
import yaml
import time

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
        svname = 'program_{}'.format(config['train_dataset'])
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

    logger = utils.Logger(file_name=os.path.join(save_path, "log_sdout.txt"), file_mode="a+", should_flush=True)

    save_reconst_samples_path = os.path.join(save_path, 'reconst_samples')

    #### Dataset ####

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    utils.log('train dataset: {} (x{})'.format(
        train_dataset[0][0].shape, len(train_dataset)))
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
            val_dataset[0][0].shape, len(val_dataset)))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    else:
        eval_val = False

    # tval
    if config.get('tval_dataset'):
        eval_tval = True
        tval_dataset = datasets.make(config['tval_dataset'],
                                     **config['tval_dataset_args'])
        tval_loader = DataLoader(tval_dataset, config['batch_size'],
                                 num_workers=8, pin_memory=True, drop_last=True)
        utils.log('tval dataset: {} (x{})'.format(
            tval_dataset[0][0].shape, len(tval_dataset)))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
    else:
        eval_tval = False

    # few-shot eval
    if config.get('fs_dataset'):
        ef_epoch = config.get('eval_fs_epoch')
        if ef_epoch is None:
            ef_epoch = 5
        eval_fs = True

        fs_dataset_tval = datasets.make(config['fs_dataset'],
                                        **config['fs_dataset_args'])
        utils.log('fs dataset_tval: {} (x{})'.format(
            fs_dataset_tval[0][0].shape, len(fs_dataset_tval)))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(fs_dataset_tval, 'dataset_tval', writer)

        fs_dataset_val = datasets.make(config['fv_dataset'],
                                       **config['fv_dataset_args'])
        utils.log('fs dataset_val: {} (x{})'.format(
            fs_dataset_val[0][0].shape, len(fs_dataset_val)))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(fs_dataset_val, 'dataset_val', writer)

        fs_dataset_train = datasets.make(config['ft_dataset'],
                                         **config['ft_dataset_args'])
        utils.log('fs dataset_train: {} (x{})'.format(
            fs_dataset_train[0][0].shape, len(fs_dataset_train)))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(fs_dataset_train, 'dataset_val', writer)

        fs_datasets = [fs_dataset_train, fs_dataset_tval, fs_dataset_val]
        ep_per_batch = 8
        n_batches = [1000 // ep_per_batch, 1000 // ep_per_batch, 1000 // ep_per_batch]
        fs_types = ['train', 'tval', 'val']

        n_way = 2
        n_query = 1
        n_shots = [6, 6, 6]
        fs_loaders = []
        for i, n_shot in enumerate(n_shots):
            fs_sampler = BongardSampler(
                fs_datasets[i].n_tasks, n_batch=n_batches[i],
                ep_per_batch=ep_per_batch, seed=random_state.randint(2 ** 31))
            fs_loader = DataLoader(fs_datasets[i], batch_sampler=fs_sampler,
                                   num_workers=8, pin_memory=True)
            fs_loaders.append(fs_loader)
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

    if config.get('_parallel'):
        sample_fn = model.module.sample
    else:
        sample_fn = model.sample

    ########

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va_base = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    for epoch in range(1, max_epoch + 1 + 1):

        timer_epoch.s()
        aves_keys = ['tl', 'ta-base_idx', 'ta-base_type', 'ta-args0', 'ta-args1',
                     'vl', 'va-base_idx', 'va-base_type', 'va-args0', 'va-args1',
                     'tvl', 'tva-base_idx', 'tva-base_type', 'tva-args0', 'tva-args1']
        if eval_fs:
            for i, n_shot in enumerate(n_shots):
                aves_keys += ['fsa-' + str(n_shot) + fs_types[i]]
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, program in tqdm(train_loader, desc='train', position=0, leave=True):
            loss, acc_base_idx_ave, acc_base_type_ave, acc_args0_ave, acc_args1_ave = \
                model(x=data.cuda(), program=program.cuda())

            loss = loss.mean()
            acc_base_idx_ave = acc_base_idx_ave.mean()
            acc_base_type_ave = acc_base_type_ave.mean()
            acc_args0_ave = acc_args0_ave.mean()
            acc_args1_ave = acc_args1_ave.mean()
            # print('loss: ', loss.item(), 'loss size: ', loss.size())

            optimizer.zero_grad()
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # todo: tuning the clip max_val
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta-base_idx'].add(acc_base_idx_ave.item())
            aves['ta-base_type'].add(acc_base_type_ave.item())
            aves['ta-args0'].add(acc_args0_ave.item())
            aves['ta-args1'].add(acc_args1_ave.item())

        utils.visualize_reconstructions(train_dataset, sample_fn, 'train_{}'.format(epoch),
                                        save_path=save_reconst_samples_path)

        # val
        if eval_val:
            model.eval()
            for data, program in tqdm(val_loader, desc='val', leave=False):
                with torch.no_grad():
                    loss, acc_base_idx_ave, acc_base_type_ave, acc_args0_ave, acc_args1_ave = \
                        model(x=data.cuda(), program=program.cuda())

                    loss = loss.mean()
                    acc_base_idx_ave = acc_base_idx_ave.mean()
                    acc_base_type_ave = acc_base_type_ave.mean()
                    acc_args0_ave = acc_args0_ave.mean()
                    acc_args1_ave = acc_args1_ave.mean()

                aves['vl'].add(loss.item())
                aves['va-base_idx'].add(acc_base_idx_ave.item())
                aves['va-base_type'].add(acc_base_type_ave.item())
                aves['va-args0'].add(acc_args0_ave.item())
                aves['va-args1'].add(acc_args1_ave.item())

            utils.visualize_reconstructions(val_dataset, sample_fn, 'val_{}'.format(epoch),
                                            save_path=save_reconst_samples_path)

        # tval
        if eval_tval:
            model.eval()
            for data, program in tqdm(tval_loader, desc='tval', leave=False):
                with torch.no_grad():
                    loss, acc_base_idx_ave, acc_base_type_ave, acc_args0_ave, acc_args1_ave = \
                        model(x=data.cuda(), program=program.cuda())

                    loss = loss.mean()
                    acc_base_idx_ave = acc_base_idx_ave.mean()
                    acc_base_type_ave = acc_base_type_ave.mean()
                    acc_args0_ave = acc_args0_ave.mean()
                    acc_args1_ave = acc_args1_ave.mean()

                aves['tvl'].add(loss.item())
                aves['tva-base_idx'].add(acc_base_idx_ave.item())
                aves['tva-base_type'].add(acc_base_type_ave.item())
                aves['tva-args0'].add(acc_args0_ave.item())
                aves['tva-args1'].add(acc_args1_ave.item())

            utils.visualize_reconstructions(tval_dataset, sample_fn, 'tval_{}'.format(epoch),
                                            save_path=save_reconst_samples_path)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            fs_model.eval()
            for i, n_shot in enumerate(n_shots):
                np.random.seed(0)
                for data, _ in tqdm(fs_loaders[i], desc='fs-' + str(n_shot), leave=False):
                    x_shot, x_query = fs.split_shot_query(
                        data.cuda(), n_way, n_shot, n_query, ep_per_batch=ep_per_batch)
                    label_query = fs.make_nk_label(
                        n_way, n_query, ep_per_batch=ep_per_batch).cuda()
                    with torch.no_grad():
                        logits = fs_model(x_shot, x_query).view(-1, n_way)
                        acc = utils.compute_acc(logits, label_query)
                    aves['fsa-' + str(n_shot) + fs_types[i]].add(acc)

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
        log_str = 'epoch {}, train {:.4f}|({:.4f}, {:.4f}, {:.4f}, {:.4f})'.format(
            epoch_str, aves['tl'], aves['ta-base_idx'], aves['ta-base_type'], aves['ta-args0'], aves['ta-args1'])

        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc-base_idx', {'train': aves['ta-base_idx']}, epoch)
        writer.add_scalars('acc-base_type', {'train': aves['ta-base_type']}, epoch)
        writer.add_scalars('acc-args0', {'train': aves['ta-args0']}, epoch)
        writer.add_scalars('acc-args1', {'train': aves['ta-args1']}, epoch)

        if eval_val:
            log_str += ', val {:.4f}|({:.4f}, {:.4f}, {:.4f}, {:.4f}), ' \
                       'tval {:.4f}|({:.4f}, {:.4f}, {:.4f}, {:.4f})'.format(
                aves['vl'], aves['va-base_idx'], aves['va-base_type'], aves['va-args0'], aves['va-args1'],
                aves['tvl'], aves['tva-base_idx'], aves['tva-base_type'], aves['tva-args0'], aves['tva-args1'])

            writer.add_scalars('loss', {'val': aves['vl']}, epoch)
            writer.add_scalars('acc-base_idx', {'val': aves['va-base_idx']}, epoch)
            writer.add_scalars('acc-base_type', {'val': aves['va-base_type']}, epoch)
            writer.add_scalars('acc-args0', {'val': aves['va-args0']}, epoch)
            writer.add_scalars('acc-args1', {'val': aves['va-args1']}, epoch)

            writer.add_scalars('loss', {'tval': aves['tvl']}, epoch)
            writer.add_scalars('acc-base_idx', {'tval': aves['tva-base_idx']}, epoch)
            writer.add_scalars('acc-base_type', {'tval': aves['tva-base_type']}, epoch)
            writer.add_scalars('acc-args0', {'tval': aves['tva-args0']}, epoch)
            writer.add_scalars('acc-args1', {'tval': aves['tva-args1']}, epoch)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            log_str += ', fs'
            for i, n_shot in enumerate(n_shots):
                key = 'fsa-' + str(n_shot) + fs_types[i]
                log_str += ' {} {}: {:.4f}'.format(n_shot, fs_types[i], aves[key])
                writer.add_scalars('acc', {key: aves[key]}, epoch)

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

            cur_va_acc = aves['va-base_idx'] + aves['va-base_type'] + aves['va-args0'] + aves['va-args1']
            if cur_va_acc > max_va_base:
                max_va_base = cur_va_acc
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
