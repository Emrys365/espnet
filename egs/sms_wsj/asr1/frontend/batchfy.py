#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import itertools
import json
import logging
import math
# matplotlib related
import os
import shutil
import tempfile

# chainer related
import chainer

from chainer import training
from chainer.training import extension

from chainer.serializers.npz import DictionarySerializer
from chainer.serializers.npz import NpzDeserializer

# io related
import numpy as np
import torch


# * -------------------- training iterator related -------------------- *
def make_batchset(data, batch_size, max_length_in, max_length_out,
                  num_batches=0, min_batch_size=1, shortest_first=False):
    """Make batch set from json dictionary

    if utts have "category" value,

        >>> data = {'utt1': {'category': 'A', 'input': ...},
        ...         'utt2': {'category': 'B', 'input': ...},
        ...         'utt3': {'category': 'B', 'input': ...},
        ...         'utt4': {'category': 'A', 'input': ...}}
        >>> make_batchset(data, batchsize=2, ...)
        [[('utt1', ...), ('utt4', ...)], [('utt2', ...), ('utt3': ...)]]

    Note that if any utts doesn't have "category",
    perform as same as "make_batchset_within_category"

    :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse
    :return: List[List[Tuple[str, dict]]] list of batches
    """

    category2data = {}  # Dict[str, dict]
    for k, v in data.items():
        category2data.setdefault(v.get('category'), {})[k] = v

    batches_list = []  # List[List[List[Tuple[str, dict]]]]
    for _, d in category2data.items():
        # batch: List[List[Tuple[str, dict]]]
        batches = make_batchset_within_category(
            data=d,
            batch_size=batch_size,
            max_length_in=max_length_in,
            max_length_out=max_length_out,
            min_batch_size=min_batch_size,
            shortest_first=shortest_first)
        batches_list.append(batches)

    if len(batches_list) == 1:
        batches = batches_list[0]
    else:
        # Concat list. This way is faster than "sum(batch_list, [])"
        batches = list(itertools.chain(*batches_list))

    # for debugging
    if num_batches > 0:
        batches = batches[:num_batches]
    logging.info('# minibatches: ' + str(len(batches)))

    # batch: List[List[Tuple[str, dict]]]
    return batches


def make_batchset_within_category(
        data, batch_size, max_length_in, max_length_out,
        min_batch_size=1, shortest_first=False):
    """Make batch set from json dictionary

    :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int min_batch_size: mininum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse
    :return: List[List[Tuple[str, dict]]] list of batches
    """

    # sort it by input lengths (long to short)
    sorted_data = sorted(data.items(), key=lambda data: int(
        data[1]['input'][0]['shape'][0]), reverse=not shortest_first)
    logging.info('# utts: ' + str(len(sorted_data)))

    # check #utts is more than min_batch_size
    if len(sorted_data) < min_batch_size:
        raise ValueError("#utts is less than min_batch_size.")

    # make list of minibatches
    minibatches = []
    start = 0
    while True:
        _, info = sorted_data[start]
        ilen = int(info['input'][0]['shape'][0])
        # olen = max(map(lambda x: int(x['shape'][0]), info['output']))
        factor = int(ilen / max_length_in)
        # factor = max(int(ilen / max_length_in), int(olen / max_length_out))

        # change batchsize depending on the input and output length
        # if ilen = 1000 and max_length_in = 800
        # then b = batchsize / 2
        # and max(min_batches, .) avoids batchsize = 0
        bs = max(min_batch_size, int(batch_size / (1 + factor)))
        end = min(len(sorted_data), start + bs)
        minibatch = sorted_data[start:end]
        if shortest_first:
            minibatch.reverse()

        # check each batch is more than minimum batchsize
        if len(minibatch) < min_batch_size:
            mod = min_batch_size - len(minibatch) % min_batch_size
            additional_minibatch = [sorted_data[i]
                                    for i in np.random.randint(0, start, mod)]
            if shortest_first:
                additional_minibatch.reverse()
            minibatch.extend(additional_minibatch)
        minibatches.append(minibatch)

        if end == len(sorted_data):
            break
        start = end

    # batch: List[List[Tuple[str, dict]]]
    return minibatches


def make_batchset_cl(data, batch_size, max_length_in, max_length_out,
                     num_batches=0, min_batch_size=1, shortest_first=True):
    """Make batch set from json dictionary (using curriculum learning)

    if utts have "category" value,

        >>> data = {'utt1': {'category': 'A', 'input': ...},
        ...         'utt2': {'category': 'B', 'input': ...},
        ...         'utt3': {'category': 'B', 'input': ...},
        ...         'utt4': {'category': 'A', 'input': ...}}
        >>> make_batchset(data, batchsize=2, ...)
        [[('utt1', ...), ('utt4', ...)], [('utt2', ...), ('utt3': ...)]]

    Note that if any utts doesn't have "category",
    perform as same as "make_batchset_within_category"

    :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int num_batches: # number of batches to use (for debug)
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse
    :param bool cleanest_first: Sort from batch with largest to smallest in terms of SNR if true, otherwise reverse
    :return: List[List[Tuple[str, dict]]] list of batches
    """

    cleanest_first = shortest_first

    category2data = {}  # Dict[str, dict]
    for k, v in data.items():
        category2data.setdefault(v.get('category'), {})[k] = v

    batches_list = []  # List[List[List[Tuple[str, dict]]]]
    for c, d in category2data.items():
        # batch: List[List[Tuple[str, dict]]]
        if shortest_first and cleanest_first:
            if c == "singlespeaker":
                batches = make_batchset_within_category(
                    data=d,
                    batch_size=batch_size,
                    max_length_in=max_length_in,
                    max_length_out=max_length_out,
                    min_batch_size=min_batch_size,
                    shortest_first=shortest_first)
            elif c == "multichannel" or c == "2channels":
                batches = make_batchset_within_category_cl(
                    data=d,
                    batch_size=batch_size,
                    max_length_in=max_length_in,
                    max_length_out=max_length_out,
                    min_batch_size=min_batch_size,
                    shortest_first=shortest_first,
                    cleanest_first=cleanest_first)
            else:
                raise ValueError("Unsupported category {}.".format(c))
        else:
            batches = make_batchset_within_category(
                data=d,
                batch_size=batch_size,
                max_length_in=max_length_in,
                max_length_out=max_length_out,
                min_batch_size=min_batch_size,
                shortest_first=shortest_first)
        batches_list.append(batches)

    if len(batches_list) == 1:
        batches = batches_list[0]
    else:
        # Concat list. This way is faster than "sum(batch_list, [])"
        if cleanest_first:
            # Interpolate the batches of batches_list
            batches = list(itertools.zip_longest(*batches_list))
            new_lst = []
            for i, b in enumerate(batches):
                new_lst += list(b)
            batches = list(filter(lambda x: x is not None, new_lst))
        else:
            batches = list(itertools.chain(*batches_list))

    # for debugging
    if num_batches > 0:
        batches = batches[:num_batches]
    logging.info('# minibatches: ' + str(len(batches)))

    # batch: List[List[Tuple[str, dict]]]
    return batches


def make_batchset_within_category_cl(
        data, batch_size, max_length_in, max_length_out,
        min_batch_size=1, shortest_first=False, cleanest_first=True):
    """Make batch set from json dictionary

    :param Dict[str, Dict[str, Any]] data: dictionary loaded from data.json
    :param int batch_size: batch size
    :param int max_length_in: maximum length of input to decide adaptive batch size
    :param int max_length_out: maximum length of output to decide adaptive batch size
    :param int min_batch_size: mininum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples to longest if true, otherwise reverse
    :return: List[List[Tuple[str, dict]]] list of batches
    """

    # sort it by SNR (small to large)
    sorted_data = sorted(data.items(), key=lambda data: abs(float(
        data[0].split('_')[3])), reverse=not cleanest_first)
    logging.info('# utts: ' + str(len(sorted_data)))

    # check #utts is more than min_batch_size
    if len(sorted_data) < min_batch_size:
        raise ValueError("#utts is less than min_batch_size.")

    # make list of minibatches
    minibatches = []
    start = 0
    while True:
        _, info = sorted_data[start]
        #ilen = int(info['input'][0]['shape'][0])
        #olen = max(map(lambda x: int(x['shape'][0]), info['output']))
        info_candidates = [sorted_data[i] for i in range(start, min(len(sorted_data), start+batch_size))]
        ilen = int(max([x[1]['input'][0]['shape'][0] for x in info_candidates]))
        # olen = int(max([max(map(lambda x: int(x['shape'][0]), infor[1]['output'])) for infor in info_candidates]))
        factor = int(ilen / max_length_in)
        # factor = max(int(ilen / max_length_in), int(olen / max_length_out))

        # change batchsize depending on the input and output length
        # if ilen = 1000 and max_length_in = 800
        # then b = batchsize / 2
        # and max(min_batches, .) avoids batchsize = 0
        bs = max(min_batch_size, int(batch_size / (1 + factor)))
        end = min(len(sorted_data), start + bs)
        minibatch = sorted_data[start:end]
        if shortest_first:
            minibatch.reverse()

        # check each batch is more than minimum batchsize
        if len(minibatch) < min_batch_size:
            mod = min_batch_size - len(minibatch) % min_batch_size
            additional_minibatch = [sorted_data[i]
                                    for i in np.random.randint(0, start, mod)]
            if shortest_first:
                additional_minibatch.reverse()
            minibatch.extend(additional_minibatch)
        minibatches.append(minibatch)

        if end == len(sorted_data):
            break
        start = end

    # batch: List[List[Tuple[str, dict]]]
    return minibatches
