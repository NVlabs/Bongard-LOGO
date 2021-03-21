# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        svname = 'meta_{}-{}shot'.format(
            config['train_dataset'], config['n_shot'])
        svname += '_' + config['model']
        if config['model_args'].get('encoder'):
            svname += '-' + config['model_args']['encoder']
        if config['model_args'].get('prog_synthesis'):
            svname += '-' + config['model_args']['prog_synthesis']
    svname += '-seed' + str(args.seed)
    if args.tag is not None:
        svname += '_' + args.tag

    save_path = os.path.join(args.save_dir, svname)
    utils.ensure_path(save_path, remove=False)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    logger = utils.Logger(file_name=os.path.join(save_path, "log_sdout.txt"), file_mode="a+", should_flush=True)

    #### Dataset ####

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    random_state = np.random.RandomState(args.seed)
    print('seed：', args.seed)

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {} (x{})'.format(
        train_dataset[0][0].shape, len(train_dataset)))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    train_sampler = BongardSampler(
        train_dataset.n_tasks, config['train_batches'],
        ep_per_batch, random_state.randint(2 ** 31))
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    # tvals
    tval_loaders = {}
    tval_name_ntasks_dict = {'tval': 2000, 'tval_ff': 600, 'tval_bd': 480,
                             'tval_hd_comb': 400, 'tval_hd_novel': 320}  # numbers depend on dataset
    for tval_type in tval_name_ntasks_dict.keys():
        if config.get('{}_dataset'.format(tval_type)):
            tval_dataset = datasets.make(config['{}_dataset'.format(tval_type)],
                                         **config['{}_dataset_args'.format(tval_type)])
            utils.log('{} dataset: {} (x{})'.format(
                tval_type, tval_dataset[0][0].shape, len(tval_dataset)))
            if config.get('visualize_datasets'):
                utils.visualize_dataset(tval_dataset, 'tval_ff_dataset', writer)
            tval_sampler = BongardSampler(
                tval_dataset.n_tasks, n_batch=tval_name_ntasks_dict[tval_type] // ep_per_batch,
                ep_per_batch=ep_per_batch, seed=random_state.randint(2 ** 31))
            tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                     num_workers=8, pin_memory=True)
            tval_loaders.update({tval_type: tval_loader})
        else:
            tval_loaders.update({tval_type: None})

    # val
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{})'.format(
        val_dataset[0][0].shape, len(val_dataset)))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = BongardSampler(
        val_dataset.n_tasks, n_batch=900 // ep_per_batch,
        ep_per_batch=ep_per_batch, seed=random_state.randint(2 ** 31))
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    ########

    #### Model and optimizer ####

    if config.get('load'):
        print('loading pretrained model: ', config['load'])
        model = models.load(torch.load(config['load']))
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            print('loading pretrained encoder: ', config['load_encoder'])
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())

        if config.get('load_prog_synthesis'):
            print('loading pretrained program synthesis model: ', config['load_prog_synthesis'])
            prog_synthesis = models.load(torch.load(config['load_prog_synthesis']))
            model.prog_synthesis.load_state_dict(prog_synthesis.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

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

    aves_keys = ['tl', 'ta', 'vl', 'va']
    tval_tuple_lst = []
    for k, v in tval_loaders.items():
        if v is not None:
            loss_key = 'tvl' + k.split('tval')[-1]
            acc_key = ' tva' + k.split('tval')[-1]
            aves_keys.append(loss_key)
            aves_keys.append(acc_key)
            tval_tuple_lst.append((k, v, loss_key, acc_key))

    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, label in tqdm(train_loader, desc='train', leave=False):

            x_shot, x_query = fs.split_shot_query(
                data.cuda(), n_train_way, n_train_shot, n_query,
                ep_per_batch=ep_per_batch)
            label_query = fs.make_nk_label(
                n_train_way, n_query,
                ep_per_batch=ep_per_batch).cuda()

            if config['model'] == 'snail':  # only use one selected label_query
                query_dix = random_state.randint(n_train_way * n_query)
                label_query = label_query.view(ep_per_batch, -1)[:, query_dix]
                x_query = x_query[:, query_dix: query_dix + 1]

            if config['model'] == 'maml':  # need grad in maml
                model.zero_grad()

            logits = model(x_shot, x_query).view(-1, n_train_way)
            loss = F.cross_entropy(logits, label_query)
            acc = utils.compute_acc(logits, label_query)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None
            loss = None

        # eval
        model.eval()

        for name, loader, name_l, name_a in [('val', val_loader, 'vl', 'va')] + tval_tuple_lst:

            if config.get('{}_dataset'.format(name)) is None:
                continue

            np.random.seed(0)
            for data, _ in tqdm(loader, desc=name, leave=False):
                x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch)
                label_query = fs.make_nk_label(
                    n_way, n_query, ep_per_batch=ep_per_batch).cuda()

                if config['model'] == 'snail':  # only use one randomly selected label_query
                    query_dix = random_state.randint(n_train_way)
                    label_query = label_query.view(ep_per_batch, -1)[:, query_dix]
                    x_query = x_query[:, query_dix: query_dix + 1]

                if config['model'] == 'maml':  # need grad in maml
                    model.zero_grad()
                    logits = model(x_shot, x_query, eval=True).view(-1, n_way)
                    loss = F.cross_entropy(logits, label_query)
                    acc = utils.compute_acc(logits, label_query)
                else:
                    with torch.no_grad():
                        logits = model(x_shot, x_query, eval=True).view(-1, n_way)
                        loss = F.cross_entropy(logits, label_query)
                        acc = utils.compute_acc(logits, label_query)

                aves[name_l].add(loss.item())
                aves[name_a].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        log_str = 'epoch {}, train {:.4f}|{:.4f}, val {:.4f}|{:.4f}'.format(
            epoch, aves['tl'], aves['ta'], aves['vl'], aves['va'])
        for tval_name, _, loss_key, acc_key in tval_tuple_lst:
            log_str += ', {} {:.4f}|{:.4f}'.format(tval_name, aves[loss_key], aves[acc_key])
            writer.add_scalars('loss', {tval_name: aves[loss_key]}, epoch)
            writer.add_scalars('acc', {tval_name: aves[acc_key]}, epoch)
        log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        utils.log(log_str)

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'val': aves['vl'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'val': aves['va'],
        }, epoch)

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
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

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
