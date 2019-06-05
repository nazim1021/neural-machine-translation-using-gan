import contextlib
import sys

import torch
from torch import cuda
from torch.utils.data.dataset import Dataset
import numpy as np

import utils
from batch_generator import BatchGenerator


class DatasetProcessing(Dataset):
    def __init__(self, data, maxlen):
        # data['train/valid']: (src_data_temp, trg_data_temp, labels)
        self.data = data
        self.data_size = int(data['labels'].size(0))
        self.maxlen = maxlen

        assert self.data['src'].size(0) == self.data['trg'].size(0) \
               and self.data['trg'].size(0) == self.data['labels'].size(0)


    def __getitem__(self, index):

        assert index < self.data_size

        source = self.data['src'][index].long()
        target = self.data['trg'][index].long()
        labels = self.data['labels'][index].long()

        return {
            'source': source,
            'target': target,
            'labels': labels
        }

    def __len__(self):
        return self.data_size


    def collater(self, samples):
        return DatasetProcessing.collate(samples, self.maxlen)

    @staticmethod
    def collate(samples, maxlen):
        if len(samples) == 0:
            return {}

        def merge(key):
            return DatasetProcessing.collate_tokens([s[key] for s in samples], maxlen)

        labels = torch.LongTensor([s['labels'] for s in samples])
        src_tokens = merge('source')
        target = merge('target')

        return {
            'src_tokens': src_tokens,
            'trg_tokens': target,
            'labels': labels
        }

    @staticmethod
    def collate_tokens(values, maxlen):
        max_input_size = max(v.size(0) for v in values)
        assert max_input_size == maxlen

        res = torch.stack(values, dim=0)

        return res


def train_dataloader(dataset, batch_size=32, seed=None, epoch=1,
                     sample_without_replacement=0, sort_by_source_size=False):
    with numpy_seed(seed):
        batch_sampler = shuffled_batches_by_size(len(dataset), batch_size=batch_size, epoch=epoch,
            sample=sample_without_replacement, sort_by_source_size=sort_by_source_size)

    return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collater, batch_sampler=batch_sampler)


def eval_dataloader(dataset, num_workers=0, batch_size=32):
    batch_sampler = batches_by_order(len(dataset), batch_size)

    return torch.utils.data.DataLoader(dataset, num_workers=num_workers, collate_fn=dataset.collater, batch_sampler=batch_sampler)

def _make_batches(indices, batch_size):
    batch = []

    for idx in map(int, indices):
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(idx)

    if len(batch) > 0:
        yield batch


def batches_by_order(data_size, batch_size=32):
    """Returns batches of indices sorted by size. Sequences with different
    source lengths are not allowed in the same batch."""

    indices = np.arange(data_size)

    return list(_make_batches(indices, batch_size))


def shuffled_batches_by_size(data_size, batch_size=32, epoch=1, sample=0, sort_by_source_size=False):
    """Returns batches of indices, bucketed by size and then shuffled. Batches
    may contain sequences of different lengths."""
    if sample:
        indices = np.random.choice(data_size, sample, replace=False)
    else:
        indices = np.random.permutation(data_size)

    batches = list(_make_batches(indices, batch_size))

    if not sort_by_source_size:
        np.random.shuffle(batches)

    return batches


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def prepare_training_data(args, dataset, split, generator, epoch_i, use_cuda):

    translator = BatchGenerator(
        generator, beam_size=args.beam, stop_early=(not args.no_early_stop),
        normalize_scores=(not args.unnormalized), len_penalty=args.lenpen,
        unk_penalty=args.unkpen)

    seed = args.seed + epoch_i
    torch.manual_seed(seed)

    # prepare trainng data
    # we must use a fixed max sentence length
    if split == 'train':
        max_positions_train = (args.fixed_max_len, args.fixed_max_len)

        itr = dataset.train_dataloader(
            'train',
            max_tokens=args.max_tokens,
            max_sentences=args.prepare_dis_batch_size,
            max_positions=max_positions_train,
            seed=seed,
            epoch=epoch_i,
            sample_without_replacement=args.sample_without_replacement,
            sort_by_source_size=(epoch_i <= args.curriculum),
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )
    else:
        max_positions_valid = (args.fixed_max_len, args.fixed_max_len)

        itr = dataset.eval_dataloader(
            'valid',
            max_tokens=args.max_tokens,
            max_sentences=args.prepare_dis_batch_size,
            max_positions=max_positions_valid,
            skip_invalid_size_inputs_valid_test=True,
            descending=True,  # largest batch first to warm the caching allocator
            shard_id=args.distributed_rank,
            num_shards=args.distributed_world_size,
        )

    src_data_temp = []
    trg_data_temp = []
    labels_temp = []
    print("preparing discriminator {0} data...".format(split))

    with torch.no_grad():
        for i, sample in enumerate(itr):
            sys.stdout.write('\r' + 'Finishing ' + str(i + 1) + '/' + str(len(itr)))
            sys.stdout.flush()

            if use_cuda:
                # wrap input tensors in cuda tensors
                sample = utils.make_variable(sample, cuda=cuda)

            # a tensor with max possible translation length
            neg_tokens = translator.generate_translation_tokens(args, dataset, sample, beam_size=args.beam, maxlen_a=args.max_len_a,
                                                                maxlen_b=args.max_len_b, nbest=args.nbest)

            # mask the results that exceeds fixed max length, and truncate at the max length
            selected_row = (neg_tokens[:, args.fixed_max_len] == dataset.dst_dict.pad()).nonzero().squeeze(1)

            if selected_row.size(0) != neg_tokens.size(0):
                print('\r' + "Warning, {0} sentences are removed due to exceeding length".format(int(neg_tokens.size(0) - selected_row.size(0))))

            neg_tokens = neg_tokens[selected_row]
            neg_tokens = neg_tokens[:, : args.fixed_max_len]

            pos_tokens = sample['target'][selected_row]

            src_tokens = sample['net_input']['src_tokens'][selected_row]

            assert neg_tokens.size() == pos_tokens.size()
            assert src_tokens.size() == pos_tokens.size()

            src_data_temp.append(src_tokens)
            trg_data_temp.append(pos_tokens)
            labels_temp.extend([1] * int(pos_tokens.size(0)))

            src_data_temp.append(src_tokens)
            trg_data_temp.append(neg_tokens)
            labels_temp.extend([0] * int(neg_tokens.size(0)))

        src_data_temp = torch.cat(src_data_temp, dim=0)
        trg_data_temp = torch.cat(trg_data_temp, dim=0)
        src_data_temp = src_data_temp.cpu().int()
        trg_data_temp = trg_data_temp.cpu().int()

        labels_temp = np.asarray(labels_temp)
        labels = torch.from_numpy(labels_temp)
        labels = labels.cpu().int()

        data = {'src': src_data_temp, 'trg': trg_data_temp, 'labels': labels}

        print('\n' + "preparing discriminator {0} data done!".format(split))

    return data