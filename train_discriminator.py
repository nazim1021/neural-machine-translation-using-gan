import logging
import os
import math
import numpy as np
from collections import OrderedDict

import torch
from torch import cuda
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset

import utils
from meters import AverageMeter
from discriminator import Discriminator
from generator import LSTMModel
from disc_dataloader import DatasetProcessing, prepare_training_data
from disc_dataloader import train_dataloader, eval_dataloader


def train_d(args, dataset):
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

    use_cuda = (torch.cuda.device_count() >= 1)

    # check checkpoints saving path
    if not os.path.exists('checkpoints/discriminator'):
        os.makedirs('checkpoints/discriminator')

    checkpoints_path = 'checkpoints/discriminator/'

    logging_meters = OrderedDict()
    logging_meters['train_loss'] = AverageMeter()
    logging_meters['train_acc'] = AverageMeter()
    logging_meters['valid_loss'] = AverageMeter()
    logging_meters['valid_acc']  = AverageMeter()
    logging_meters['update_times'] = AverageMeter()

    # Build model
    discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict,  use_cuda=use_cuda)

    # Load generator
    assert os.path.exists('checkpoints/generator/best_gmodel.pt')
    generator = LSTMModel(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    model_dict = generator.state_dict()
    pretrained_dict = torch.load('checkpoints/generator/best_gmodel.pt')
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    generator.load_state_dict(model_dict)

    if use_cuda:
        if torch.cuda.device_count() > 1:
            discriminator = torch.nn.DataParallel(discriminator).cuda()
            # generator = torch.nn.DataParallel(generator).cuda()
            generator.cuda()
        else:
            generator.cuda()
            discriminator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()

    criterion = torch.nn.CrossEntropyLoss()

    # optimizer = eval("torch.optim." + args.d_optimizer)(filter(lambda x: x.requires_grad, discriminator.parameters()),
    #                                                     args.d_learning_rate, momentum=args.momentum, nesterov=True)

    optimizer = torch.optim.RMSprop(filter(lambda x: x.requires_grad, discriminator.parameters()), 1e-4)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=args.lr_shrink)

    # Train until the accuracy achieve the define value
    max_epoch = args.max_epoch or math.inf
    epoch_i = 1
    trg_acc = 0.82
    best_dev_loss = math.inf
    lr = optimizer.param_groups[0]['lr']

    # validation set data loader (only prepare once)
    train = prepare_training_data(args, dataset, 'train', generator, epoch_i, use_cuda)
    valid = prepare_training_data(args, dataset, 'valid', generator, epoch_i, use_cuda)
    data_train = DatasetProcessing(data=train, maxlen=args.fixed_max_len)
    data_valid = DatasetProcessing(data=valid, maxlen=args.fixed_max_len)

    # main training loop
    while lr > args.min_d_lr and epoch_i <= max_epoch:
        logging.info("At {0}-th epoch.".format(epoch_i))

        seed = args.seed + epoch_i
        torch.manual_seed(seed)


        if args.sample_without_replacement > 0 and epoch_i > 1:
            train = prepare_training_data(args, dataset, 'train', generator, epoch_i, use_cuda)
            data_train = DatasetProcessing(data=train, maxlen=args.fixed_max_len)


        # discriminator training dataloader
        train_loader = train_dataloader(data_train, batch_size=args.joint_batch_size,
                                        seed=seed, epoch=epoch_i, sort_by_source_size=False)

        valid_loader = eval_dataloader(data_valid, num_workers=4, batch_size=args.joint_batch_size)

        # set training mode
        discriminator.train()

        # reset meters
        for key, val in logging_meters.items():
            if val is not None:
                val.reset()

        for i, sample in enumerate(train_loader):
            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=use_cuda)

            disc_out = discriminator(sample['src_tokens'], sample['trg_tokens'])

            loss = criterion(disc_out, sample['labels'])
            _, prediction = F.softmax(disc_out, dim=1).topk(1)
            acc = torch.sum(prediction == sample['labels'].unsqueeze(1)).float() / len(sample['labels'])

            logging_meters['train_acc'].update(acc.item())
            logging_meters['train_loss'].update(loss.item())
            logging.debug("D training loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ". \
                          format(logging_meters['train_loss'].avg, acc, logging_meters['train_acc'].avg,
                                 optimizer.param_groups[0]['lr'], i))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(discriminator.parameters(), args.clip_norm)
            optimizer.step()

            # del src_tokens, trg_tokens, loss, disc_out, labels, prediction, acc
            del disc_out, loss, prediction, acc


        # set validation mode
        discriminator.eval()

        for i, sample in enumerate(valid_loader):
            with torch.no_grad():
                if use_cuda:
                    # wrap input tensors in cuda tensors
                    sample = utils.make_variable(sample, cuda=use_cuda)

                disc_out = discriminator(sample['src_tokens'], sample['trg_tokens'])

                loss = criterion(disc_out, sample['labels'])
                _, prediction = F.softmax(disc_out, dim=1).topk(1)
                acc = torch.sum(prediction == sample['labels'].unsqueeze(1)).float() / len(sample['labels'])

                logging_meters['valid_acc'].update(acc.item())
                logging_meters['valid_loss'].update(loss.item())
                logging.debug("D eval loss {0:.3f}, acc {1:.3f}, avgAcc {2:.3f}, lr={3} at batch {4}: ". \
                              format(logging_meters['valid_loss'].avg, acc, logging_meters['valid_acc'].avg,
                                     optimizer.param_groups[0]['lr'], i))

            del disc_out, loss, prediction, acc

        lr_scheduler.step(logging_meters['valid_loss'].avg)

        if logging_meters['valid_acc'].avg >= 0.70:
            torch.save(discriminator.state_dict(), checkpoints_path + "ce_{0:.3f}_acc_{1:.3f}.epoch_{2}.pt" \
                       .format(logging_meters['valid_loss'].avg, logging_meters['valid_acc'].avg, epoch_i))

            if logging_meters['valid_loss'].avg < best_dev_loss:
                best_dev_loss = logging_meters['valid_loss'].avg
                torch.save(discriminator.state_dict(), checkpoints_path + "best_dmodel.pt")

        # pretrain the discriminator to achieve accuracy 82%
        if logging_meters['valid_acc'].avg >= trg_acc:
            return

        epoch_i += 1
