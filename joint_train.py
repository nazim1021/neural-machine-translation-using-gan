import argparse
import logging
import math
import dill
import os
import options
import random
import numpy as np
from collections import OrderedDict

import torch
from torch import cuda
from torch.autograd import Variable

import data
import utils
from meters import AverageMeter
from discriminator import Discriminator
from generator import LSTMModel
from train_generator import train_g
from train_discriminator import train_d
from PGLoss import PGLoss



logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Adversarial-NMT.")

# Load args
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_distributed_training_args(parser)
options.add_optimization_args(parser)
options.add_checkpoint_args(parser)
options.add_generator_model_args(parser)
options.add_discriminator_model_args(parser)
options.add_generation_args(parser)

def main(args):
    use_cuda = (len(args.gpuid) >= 1)
    print("{0} GPU(s) are available".format(cuda.device_count()))

    # Load dataset
    splits = ['train', 'valid']
    if data.has_binary_files(args.data, splits):
        dataset = data.load_dataset(
            args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
    else:
        dataset = data.load_raw_text_dataset(
            args.data, splits, args.src_lang, args.trg_lang, args.fixed_max_len)
    if args.src_lang is None or args.trg_lang is None:
        # record inferred languages in args, so that it's saved in checkpoints
        args.src_lang, args.trg_lang = dataset.src, dataset.dst

    print('| [{}] dictionary: {} types'.format(dataset.src, len(dataset.src_dict)))
    print('| [{}] dictionary: {} types'.format(dataset.dst, len(dataset.dst_dict)))
    
    for split in splits:
        print('| {} {} {} examples'.format(args.data, split, len(dataset.splits[split])))
    
    g_logging_meters = OrderedDict()
    g_logging_meters['train_loss'] = AverageMeter()
    g_logging_meters['valid_loss'] = AverageMeter()
    g_logging_meters['train_acc'] = AverageMeter()
    g_logging_meters['valid_acc'] = AverageMeter()
    g_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    d_logging_meters = OrderedDict()
    d_logging_meters['train_loss'] = AverageMeter()
    d_logging_meters['valid_loss'] = AverageMeter()
    d_logging_meters['train_acc'] = AverageMeter()
    d_logging_meters['valid_acc'] = AverageMeter()
    d_logging_meters['bsz'] = AverageMeter()  # sentences per batch

    # Set model parameters
    args.encoder_embed_dim = 1000
    args.encoder_layers = 2 # 4
    args.encoder_dropout_out = 0
    args.decoder_embed_dim = 1000
    args.decoder_layers = 2 # 4
    args.decoder_out_embed_dim = 1000
    args.decoder_dropout_out = 0
    args.bidirectional = False

    generator = LSTMModel(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    print("Generator loaded successfully!")
    discriminator = Discriminator(args, dataset.src_dict, dataset.dst_dict, use_cuda=use_cuda)
    print("Discriminator loaded successfully!")

    
    if use_cuda:
        if torch.cuda.device_count() > 1:
            discriminator = torch.nn.DataParallel(discriminator).cuda()
            generator = torch.nn.DataParallel(generator).cuda()
        else:
            generator.cuda()
            discriminator.cuda()
    else:
        discriminator.cpu()
        generator.cpu()

    # adversarial training checkpoints saving path
    if not os.path.exists('checkpoints/joint'):
        os.makedirs('checkpoints/joint')
    checkpoints_path = 'checkpoints/joint/'

    # define loss function
    g_criterion = torch.nn.NLLLoss(ignore_index=dataset.dst_dict.pad(),reduction='sum')
    d_criterion = torch.nn.BCELoss()
    pg_criterion = PGLoss(ignore_index=dataset.dst_dict.pad(), size_average=True,reduce=True)

    # fix discriminator word embedding (as Wu et al. do)
    for p in discriminator.embed_src_tokens.parameters():
        p.requires_grad = False
    for p in discriminator.embed_trg_tokens.parameters():
        p.requires_grad = False

    # define optimizer
    g_optimizer = eval("torch.optim." + args.g_optimizer)(filter(lambda x: x.requires_grad,
                                                                 generator.parameters()),
                                                          args.g_learning_rate)

    d_optimizer = eval("torch.optim." + args.d_optimizer)(filter(lambda x: x.requires_grad,
                                                                 discriminator.parameters()),
                                                          args.d_learning_rate,
                                                          momentum=args.momentum,
                                                          nesterov=True)

    # start joint training
    best_dev_loss = math.inf
    num_update = 0
    # main training loop
    for epoch_i in range(1, args.epochs + 1):
        logging.info("At {0}-th epoch.".format(epoch_i))

        seed = args.seed + epoch_i
        torch.manual_seed(seed)

        max_positions_train = (args.fixed_max_len, args.fixed_max_len)

        # Initialize dataloader, starting at batch_offset
        trainloader = dataset.train_dataloader(
            'train',
            max_tokens=args.max_tokens,
            max_sentences=args.joint_batch_size,
            max_positions=max_positions_train,
            # seed=seed,
            epoch=epoch_i,
            sample_without_replacement=args.sample_without_replacement,
            sort_by_source_size=(epoch_i <= args.curriculum),
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )

        # reset meters
        for key, val in g_logging_meters.items():
            if val is not None:
                val.reset()
        for key, val in d_logging_meters.items():
            if val is not None:
                val.reset()

        # set training mode
        generator.train()
        discriminator.train()
        update_learning_rate(num_update, 8e4, args.g_learning_rate, args.lr_shrink, g_optimizer)

        for i, sample in enumerate(trainloader):

            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            ## part I: use gradient policy method to train the generator

            # use policy gradient training when random.random() > 50%
            if random.random()  >= 0.5:

                print("Policy Gradient Training")
                
                sys_out_batch = generator(sample) # 64 X 50 X 6632

                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 * 50) X 6632   
                 
                _,prediction = out_batch.topk(1)
                prediction = prediction.squeeze(1) # 64*50 = 3200
                prediction = torch.reshape(prediction, sample['net_input']['src_tokens'].shape) # 64 X 50
                
                with torch.no_grad():
                    reward = discriminator(sample['net_input']['src_tokens'], prediction) # 64 X 1

                train_trg_batch = sample['target'] # 64 x 50
                
                pg_loss = pg_criterion(sys_out_batch, train_trg_batch, reward, use_cuda)
                sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens'] # 64
                logging_loss = pg_loss / math.log(2)
                g_logging_meters['train_loss'].update(logging_loss.item(), sample_size)
                logging.debug(f"G policy gradient loss at batch {i}: {pg_loss.item():.3f}, lr={g_optimizer.param_groups[0]['lr']}")
                g_optimizer.zero_grad()
                pg_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                g_optimizer.step()

            else:
                # MLE training
                print("MLE Training")

                sys_out_batch = generator(sample)

                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  

                train_trg_batch = sample['target'].view(-1) # 64*50 = 3200

                loss = g_criterion(out_batch, train_trg_batch)

                sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
                nsentences = sample['target'].size(0)
                logging_loss = loss.data / sample_size / math.log(2)
                g_logging_meters['bsz'].update(nsentences)
                g_logging_meters['train_loss'].update(logging_loss, sample_size)
                logging.debug(f"G MLE loss at batch {i}: {g_logging_meters['train_loss'].avg:.3f}, lr={g_optimizer.param_groups[0]['lr']}")
                g_optimizer.zero_grad()
                loss.backward()
                # all-reduce grads and rescale by grad_denom
                for p in generator.parameters():
                    if p.requires_grad:
                        p.grad.data.div_(sample_size)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.clip_norm)
                g_optimizer.step()

            num_update += 1


            # part II: train the discriminator
            bsz = sample['target'].size(0) # batch_size = 64
        
            src_sentence = sample['net_input']['src_tokens'] # 64 x max-len i.e 64 X 50

            # now train with machine translation output i.e generator output
            true_sentence = sample['target'].view(-1) # 64*50 = 3200
            
            true_labels = Variable(torch.ones(sample['target'].size(0)).float()) # 64 length vector

            with torch.no_grad():
                sys_out_batch = generator(sample) # 64 X 50 X 6632

            out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  
                
            _,prediction = out_batch.topk(1)
            prediction = prediction.squeeze(1)  #64 * 50 = 6632
            
            fake_labels = Variable(torch.zeros(sample['target'].size(0)).float()) # 64 length vector

            fake_sentence = torch.reshape(prediction, src_sentence.shape) # 64 X 50 

            if use_cuda:
                fake_labels = fake_labels.cuda()
            
            disc_out = discriminator(src_sentence, fake_sentence) # 64 X 1
            
            d_loss = d_criterion(disc_out.squeeze(1), fake_labels)

            acc = torch.sum(torch.round(disc_out).squeeze(1) == fake_labels).float() / len(fake_labels)

            d_logging_meters['train_acc'].update(acc)
            d_logging_meters['train_loss'].update(d_loss)
            logging.debug(f"D training loss {d_logging_meters['train_loss'].avg:.3f}, acc {d_logging_meters['train_acc'].avg:.3f} at batch {i}")
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()



        # validation
        # set validation mode
        generator.eval()
        discriminator.eval()
        # Initialize dataloader
        max_positions_valid = (args.fixed_max_len, args.fixed_max_len)
        valloader = dataset.eval_dataloader(
            'valid',
            max_tokens=args.max_tokens,
            max_sentences=args.joint_batch_size,
            max_positions=max_positions_valid,
            skip_invalid_size_inputs_valid_test=True,
            descending=True,  # largest batch first to warm the caching allocator
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )

        # reset meters
        for key, val in g_logging_meters.items():
            if val is not None:
                val.reset()
        for key, val in d_logging_meters.items():
            if val is not None:
                val.reset()

        for i, sample in enumerate(valloader):

            with torch.no_grad():
                if use_cuda:
                    # wrap input tensors in cuda tensors
                    sample = utils.make_variable(sample, cuda=cuda)

                # generator validation
                sys_out_batch = generator(sample)
                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  
                dev_trg_batch = sample['target'].view(-1) # 64*50 = 3200

                loss = g_criterion(out_batch, dev_trg_batch)
                sample_size = sample['target'].size(0) if args.sentence_avg else sample['ntokens']
                loss = loss / sample_size / math.log(2)
                g_logging_meters['valid_loss'].update(loss, sample_size)
                logging.debug(f"G dev loss at batch {i}: {g_logging_meters['valid_loss'].avg:.3f}")

                # discriminator validation
                bsz = sample['target'].size(0)
                src_sentence = sample['net_input']['src_tokens']
                # train with half human-translation and half machine translation

                true_sentence = sample['target']
                true_labels = Variable(torch.ones(sample['target'].size(0)).float())

                with torch.no_grad():
                    sys_out_batch = generator(sample)

                out_batch = sys_out_batch.contiguous().view(-1, sys_out_batch.size(-1)) # (64 X 50) X 6632  

                _,prediction = out_batch.topk(1)
                prediction = prediction.squeeze(1)  #64 * 50 = 6632

                fake_labels = Variable(torch.zeros(sample['target'].size(0)).float())

                fake_sentence = torch.reshape(prediction, src_sentence.shape) # 64 X 50 

                if use_cuda:
                    fake_labels = fake_labels.cuda()

                disc_out = discriminator(src_sentence, fake_sentence)
                d_loss = d_criterion(disc_out.squeeze(1), fake_labels)
                acc = torch.sum(torch.round(disc_out).squeeze(1) == fake_labels).float() / len(fake_labels)
                d_logging_meters['valid_acc'].update(acc)
                d_logging_meters['valid_loss'].update(d_loss)
                logging.debug(f"D dev loss {d_logging_meters['valid_loss'].avg:.3f}, acc {d_logging_meters['valid_acc'].avg:.3f} at batch {i}")

        torch.save(generator,
                   open(checkpoints_path + f"joint_{g_logging_meters['valid_loss'].avg:.3f}.epoch_{epoch_i}.pt",
                        'wb'), pickle_module=dill)

        if g_logging_meters['valid_loss'].avg < best_dev_loss:
            best_dev_loss = g_logging_meters['valid_loss'].avg
            torch.save(generator, open(checkpoints_path + "best_gmodel.pt", 'wb'), pickle_module=dill)


def update_learning_rate(update_times, target_times, init_lr, lr_shrink, optimizer):

    lr = init_lr * (lr_shrink ** (update_times // target_times))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(f"unknown arguments: {parser.parse_known_args()[1]}")
  main(options)