'''

This code is adapted from Facebook Fairseq-py
Visit https://github.com/facebookresearch/fairseq-py for more information

'''

from collections import defaultdict
import contextlib
import logging
import os
import torch
import traceback

from torch.autograd import Variable
from torch.serialization import default_restore_location


def make_variable(sample, cuda=False):
    """Wrap input tensors in Variable class."""

    if len(sample) == 0:
        return {}

    def _make_variable(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            if cuda and torch.cuda.is_available():
                maybe_tensor = maybe_tensor.cuda()
                return Variable(maybe_tensor)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _make_variable(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_make_variable(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _make_variable(sample)

def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value

