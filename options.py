# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import argparse
import torch

def add_general_args(parser):
    parser.add_argument("--seed", default=1, type=int,
                      help="Random seed. (default=1)")
    return parser

def add_dataset_args(parser):
    parser.add_argument("--data", required=True,
                        help="File prefix for training set.")
    parser.add_argument("--src_lang", default="de",
                        help="Source Language. (default = de)")
    parser.add_argument("--trg_lang", default="en",
                        help="Target Language. (default = en)")
    parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                       help='max number of tokens in the source sequence')
    parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                       help='max number of tokens in the target sequence')
    parser.add_argument('--skip-invalid-size-inputs-valid-test', default=True, type=bool,
                       help='Ignore too long or too short lines in valid and test set')
    parser.add_argument('--max-tokens', default=6000, type=int, metavar='N',
                       help='maximum number of tokens in a batch')
    parser.add_argument('--max-sentences', '--batch-size', type=int, metavar='N',
                       help='maximum number of sentences in a batch')
    parser.add_argument('--joint-batch-size', type=int, default=32, metavar='N',
                        help='batch size for joint training')
    parser.add_argument('--prepare-dis-batch-size', type=int, default=128, metavar='N',
                        help='batch size for preparing discriminator training')

    return parser

def add_distributed_training_args(parser):
    parser.add_argument('--distributed-world-size', type=int, metavar='N',
                       default=torch.cuda.device_count(),
                       help='total number of GPUs across all nodes (default: all visible GPUs)')
    parser.add_argument('--distributed-rank', default=0, type=int,
                       help='rank of the current worker')
    parser.add_argument("--gpuid", default=0, nargs='+', type=int,
                        help="ID of gpu device to use. Empty implies cpu usage.")

    return parser

def add_optimization_args(parser):
    parser.add_argument('--max-epoch', '--me', default=0, type=int, metavar='N',
                        help='force stop training at specified epoch')
    parser.add_argument("--epochs", default=12, type=int,
                        help="Epochs through the data. (default=12)")
    parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                        help="Optimizer of choice for training. (default=Adam)")
    parser.add_argument("--g_optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                        help="Optimizer of choice for training. (default=Adam)")
    parser.add_argument("--d_optimizer", default="SGD", choices=["SGD", "Adadelta", "Adam"],
                        help="Optimizer of choice for training. (default=Adam)")
    parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                        help="Learning rate of the optimization. (default=0.1)")
    parser.add_argument("--g_learning_rate", "-glr", default=1e-3, type=float,
                        help="Learning rate of the generator. (default=0.001)")
    parser.add_argument("--d_learning_rate", "-dlr", default=1e-3, type=float,
                        help="Learning rate of the discriminator. (default=0.001)")
    parser.add_argument("--lr_shrink", default=0.5, type=float,
                        help='learning rate shrink factor, lr_new = (lr * lr_shrink)')
    parser.add_argument('--min-g-lr', default=1e-5, type=float, metavar='LR',
                        help='minimum learning rate')
    parser.add_argument('--min-d-lr', default=1e-6, type=float, metavar='LR',
                        help='minimum learning rate')
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Momentum when performing SGD. (default=0.9)")
    parser.add_argument("--use_estop", default=False, type=bool,
                        help="Whether use early stopping criteria. (default=False)")
    parser.add_argument("--estop", default=1e-2, type=float,
                        help="Early stopping criteria on the development set. (default=1e-2)")
    parser.add_argument('--clip-norm', default=5.0, type=float,
                       help='clip threshold of gradients')
    parser.add_argument('--curriculum', default=0, type=int, metavar='N',
                       help='sort batches by source length for first N epochs')
    parser.add_argument('--sample-without-replacement', default=0, type=int, metavar='N',
                       help='If bigger than 0, use that number of mini-batches for each epoch,'
                            ' where each sample is drawn randomly without replacement from the'
                            ' dataset')
    parser.add_argument('--sentence-avg', action='store_true',
                       help='normalize gradients by the number of sentences in a batch'
                            ' (default is to normalize by number of tokens)')

    return parser


def add_checkpoint_args(parser):
    parser.add_argument("--model_file", help="Location to dump the models.")
    return parser

def add_generator_model_args(parser):
    parser.add_argument('--encoder-embed-dim', default=512, type=int,
                       help='encoder embedding dimension')
    parser.add_argument('--encoder-layers', default=1, type=int,
                       help='encoder layers [(dim, kernel_size), ...]')
    parser.add_argument('--decoder-embed-dim', default=512, type=int,
                       help='decoder embedding dimension')
    parser.add_argument('--decoder-layers', default=1, type=int,
                       help='decoder layers [(dim, kernel_size), ...]')
    parser.add_argument('--decoder-out-embed-dim', default=512, type=int,
                       help='decoder output dimension')
    parser.add_argument('--encoder-dropout-in', default=0.1, type=float,
                       help='dropout probability for encoder input embedding')
    parser.add_argument('--encoder-dropout-out', default=0.1, type=float,
                       help='dropout probability for encoder output')
    parser.add_argument('--decoder-dropout-in', default=0.1, type=float,
                       help='dropout probability for decoder input embedding')
    parser.add_argument('--decoder-dropout-out', default=0.1, type=float,
                       help='dropout probability for decoder output')
    parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                        help='dropout probability')
    parser.add_argument('--bidirectional', default=False, type=bool,
                       help='unidirectional or bidirectional encoder')
    return parser

def add_discriminator_model_args(parser):
    parser.add_argument('--fixed-max-len', default=50, type=int,
                       help='the max length the discriminator can hold')
    parser.add_argument('--d-sample-size', default=5000, type=int,
                       help='how many data used to pretrain d in one epoch')
    return parser

def add_generation_args(parser):
    parser.add_argument('--beam', default=5, type=int, metavar='N',
                        help='beam size')
    parser.add_argument('--nbest', default=1, type=int, metavar='N',
                        help='number of hypotheses to output')
    parser.add_argument('--max-len-a', default=0, type=float, metavar='N',
                        help=('generate sequences of maximum length ax + b, '
                              'where x is the source length'))
    parser.add_argument('--max-len-b', default=200, type=int, metavar='N',
                        help=('generate sequences of maximum length ax + b, '
                              'where x is the source length'))
    parser.add_argument('--remove-bpe', nargs='?', const='@@ ', default=None,
                        help='remove BPE tokens before scoring')
    parser.add_argument('--no-early-stop', action='store_true',
                        help=('continue searching even after finalizing k=beam '
                              'hypotheses; this is more correct, but increases '
                              'generation time by 50%%'))
    parser.add_argument('--unnormalized', action='store_true',
                        help='compare unnormalized hypothesis scores')
    parser.add_argument('--lenpen', default=1, type=float,
                        help='length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
    parser.add_argument('--unkpen', default=0, type=float,
                        help='unknown word penalty: <0 produces more unks, >0 produces fewer')
    parser.add_argument('--replace-unk', nargs='?', const=True, default=None,
                        help='perform unknown replacement (optionally with alignment dictionary)')

    return parser
