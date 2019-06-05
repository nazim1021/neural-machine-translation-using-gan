import argparse
import logging

import torch
import os
from torch import cuda
import options
import data
from generator import LSTMModel

from sequence_generator import SequenceGenerator

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

parser = argparse.ArgumentParser(
    description="Driver program for JHU Adversarial-NMT.")

# Load args
options.add_general_args(parser)
options.add_dataset_args(parser)
options.add_checkpoint_args(parser)
options.add_distributed_training_args(parser)
options.add_generation_args(parser)
options.add_generator_model_args(parser)


def main(args):

    use_cuda = (len(args.gpuid) >= 1)
    if args.gpuid:
        cuda.set_device(args.gpuid[0])

        # Load dataset
        if args.replace_unk is None:
            dataset = data.load_dataset(
                args.data,
                ['test'],
                args.src_lang,
                args.trg_lang,
            )
        else:
            dataset = data.load_raw_text_dataset(
                args.data,
                ['test'],
                args.src_lang,
                args.trg_lang,
            )

        if args.src_lang is None or args.trg_lang is None:
            # record inferred languages in args, so that it's saved in checkpoints
            args.src_lang, args.trg_lang = dataset.src, dataset.dst

        print('| [{}] dictionary: {} types'.format(
            dataset.src, len(dataset.src_dict)))
        print('| [{}] dictionary: {} types'.format(
            dataset.dst, len(dataset.dst_dict)))
        print('| {} {} {} examples'.format(
            args.data, 'test', len(dataset.splits['test'])))

    # Set model parameters
    # args.encoder_embed_dim = 1000
    # args.encoder_layers = 1
    # args.encoder_dropout_out = 0
    # args.decoder_embed_dim = 1000
    # args.decoder_layers = 2
    # args.decoder_out_embed_dim = 1000
    # args.decoder_dropout_out = 0
    # args.bidirectional = False

    # Load model
    g_model_path = 'checkpoints/generator/best_gmodel.pt'
    assert os.path.exists(g_model_path)
    generator = LSTMModel(args, dataset.src_dict,
                          dataset.dst_dict, use_cuda=use_cuda)
    model_dict = generator.state_dict()
    model = torch.load(g_model_path)
    pretrained_dict = model #.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    generator.load_state_dict(model_dict)
    generator.eval()

    print("Generator loaded successfully!")

    if use_cuda > 0:
        generator.cuda()
    else:
        generator.cpu()

    max_positions = generator.encoder.max_positions()

    testloader = dataset.eval_dataloader(
        'test',
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        skip_invalid_size_inputs_valid_test=args.skip_invalid_size_inputs_valid_test,
    )

    translator = SequenceGenerator(
        generator, beam_size=args.beam, stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
        unk_penalty=args.unkpen)

    if use_cuda:
        translator.cuda()

    with open('predictions.txt', 'wb') as translation_writer:
        with open('real.txt', 'wb') as ground_truth_writer:

            translations = translator.generate_batched_itr(
                testloader, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b, cuda=use_cuda)

            for sample_id, src_tokens, target_tokens, hypos in translations:
                # Process input and ground truth
                target_tokens = target_tokens.int().cpu()
                src_str = dataset.src_dict.string(src_tokens, args.remove_bpe)
                target_str = dataset.dst_dict.string(
                    target_tokens, args.remove_bpe, escape_unk=True)

                # Process top predictions
                for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                    hypo_tokens = hypo['tokens'].int().cpu()
                    hypo_str = dataset.dst_dict.string(
                        hypo_tokens, args.remove_bpe)

                    hypo_str += '\n'
                    target_str += '\n'

                    translation_writer.write(hypo_str.encode('utf-8'))
                    ground_truth_writer.write(target_str.encode('utf-8'))


if __name__ == "__main__":
    ret = parser.parse_known_args()
    args = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(
            parser.parse_known_args()[1]))
    main(args)
