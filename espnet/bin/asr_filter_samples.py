#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Automatic speech recognition model training script."""

import logging
import multiprocessing as mp
import os
import random
import subprocess
import sys

import configargparse
import numpy as np

from espnet.utils.cli_utils import strtobool
from espnet.utils.training.batchfy import BATCH_COUNT_CHOICES
from espnet.bin.asr_train import get_parser


def main(cmd_args):
    parser = get_parser()
    parser.add_argument('--output-json-path', type=str, required=True,
                        help='output path of the filtered data.json')
    args, _ = parser.parse_known_args(cmd_args)
    if args.backend == "chainer" and args.train_dtype != "float32":
        raise NotImplementedError(
            f"chainer backend does not support --train-dtype {args.train_dtype}."
            "Use --dtype float32.")
    if args.ngpu == 0 and args.train_dtype in ("O0", "O1", "O2", "O3", "float16"):
        raise ValueError(f"--train-dtype {args.train_dtype} does not support the CPU backend.")

    from espnet.utils.dynamic_import import dynamic_import
    if args.model_module is None:
        model_module = "espnet.nets." + args.backend + "_backend.e2e_asr:E2E"
    else:
        model_module = args.model_module
    model_class = dynamic_import(model_module)
    model_class.add_arguments(parser)

    args = parser.parse_args(cmd_args)
    args.model_module = model_module
    if 'chainer_backend' in args.model_module:
        args.backend = 'chainer'
    if 'pytorch_backend' in args.model_module:
        args.backend = 'pytorch'

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # If --ngpu is not given,
    #   1. if CUDA_VISIBLE_DEVICES is set, all visible devices
    #   2. if nvidia-smi exists, use all devices
    #   3. else ngpu=0
    if args.ngpu is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            ngpu = len(cvd.split(','))
        else:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
            try:
                p = subprocess.run(['nvidia-smi', '-L'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            except (subprocess.CalledProcessError, FileNotFoundError):
                ngpu = 0
            else:
                ngpu = len(p.stderr.decode().split('\n')) - 1
    else:
        ngpu = args.ngpu
    logging.info(f"ngpu: {ngpu}")

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dictionary for debug log
    if args.dict is not None:
        with open(args.dict, 'rb') as f:
            dictionary = f.readlines()
        char_list = [entry.decode('utf-8').split(' ')[0]
                     for entry in dictionary]
        char_list.insert(0, '<blank>')
        char_list.append('<eos>')
        args.char_list = char_list
    else:
        args.char_list = None

    # train
    logging.info('backend = ' + args.backend)

    if args.num_spkrs == 1:
#        if args.backend == "chainer":
#            from espnet.asr.chainer_backend.asr import train
#            train(args)
#        elif args.backend == "pytorch":
#            from espnet.asr.pytorch_backend.asr import train
#            train(args)
#        else:
        raise ValueError("Only pytorch are supported.")
    else:
        # FIXME(kamo): Support --model-module
        if args.backend == "pytorch":
            logging.warning('Filtering Samples for Multi-Speaker ASR training')
            from espnet.asr.pytorch_backend.asr_mix_filter_samples import pse_train
            pse_train(args)
        else:
            raise ValueError("Only pytorch is supported.")


if __name__ == '__main__':
    # NOTE(kan-bayashi): setting multiple times causes RuntimeError
    #   See also https://github.com/pytorch/pytorch/issues/3492
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main(sys.argv[1:])
