#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Frontend model training script."""

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


# NOTE: you need this func to generate our sphinx doc
def get_parser(parser=None, required=True):
    parser = configargparse.ArgumentParser(
        description='Transcribe text from speech using a speech recognition model on one CPU or GPU',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    # general configuration
    parser.add('--config', is_config_file=True,
               help='Config file path')
    parser.add('--config2', is_config_file=True,
               help='Second config file path that overwrites the settings in `--config`')
    parser.add('--config3', is_config_file=True,
               help='Third config file path that overwrites the settings in `--config` and `--config2`')

    parser.add_argument('--ngpu', type=int, default=0,
                        help='Number of GPUs')
    parser.add_argument('--dtype', choices=("float16", "float32", "float64"), default="float32",
                        help='Float precision (only available in --api v2)')
    parser.add_argument('--backend', type=str, default='pytorch',
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--debugmode', type=int, default=1,
                        help='Debugmode')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--verbose', '-V', type=int, default=1,
                        help='Verbose option')
    parser.add_argument('--batchsize', type=int, default=1,
                        help='Batch size for beam search (0: means no batch processing)')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    parser.add_argument('--api', default="v1", choices=["v1", "v2"],
                        help='''Beam search APIs
v1: Default API. It only supports the ASRInterface.recognize method and DefaultRNNLM.
v2: Experimental API. It supports any models that implements ScorerInterface.''')
    # task related
    parser.add_argument('--recog-json', type=str,
                        help='Filename of recognition data (json)')
    parser.add_argument('--result-label', type=str, default='',
                        help='Filename of result label data (json)')
    # model (parameter) related
    parser.add_argument('--model', type=str, required=True,
                        help='Model file parameters to read')
    parser.add_argument('--model-conf', type=str, default=None,
                        help='Model config file')
    parser.add_argument('--num-spkrs', type=int, default=1,
                        choices=[1, 2],
                        help='Number of speakers in the speech')
    parser.add_argument('--num-encs', default=1, type=int,
                        help='Number of encoders in the model.')
    # search related
    parser.add_argument('--nbest', type=int, default=1,
                        help='Output N-best hypotheses')
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size')
    parser.add_argument('--penalty', type=float, default=0.0,
                        help='Incertion penalty')
    parser.add_argument('--maxlenratio', type=float, default=0.0,
                        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""")
    parser.add_argument('--minlenratio', type=float, default=0.0,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--ctc-weight', type=float, default=0.0,
                        help='CTC weight in joint decoding')
    parser.add_argument('--weights-ctc-dec', type=float, action='append',
                        help='ctc weight assigned to each encoder during decoding.[in multi-encoder mode only]')
    parser.add_argument('--ctc-window-margin', type=int, default=0,
                        help="""Use CTC window with margin parameter to accelerate
                        CTC/attention decoding especially on GPU. Smaller magin
                        makes decoding faster, but may increase search errors.
                        If margin=0 (default), this function is disabled""")
    # transducer related
    parser.add_argument('--score-norm-transducer', type=strtobool, nargs='?',
                        default=True,
                        help='Normalize transducer scores by length')
    # rnnlm related
    parser.add_argument('--rnnlm', type=str, default=None,
                        help='RNNLM model file to read')
    parser.add_argument('--rnnlm-conf', type=str, default=None,
                        help='RNNLM model config file to read')
    parser.add_argument('--word-rnnlm', type=str, default=None,
                        help='Word RNNLM model file to read')
    parser.add_argument('--word-rnnlm-conf', type=str, default=None,
                        help='Word RNNLM model config file to read')
    parser.add_argument('--word-dict', type=str, default=None,
                        help='Word list to read')
    parser.add_argument('--lm-weight', type=float, default=0.1,
                        help='RNNLM weight')
    # streaming related
    parser.add_argument('--streaming-mode', type=str, default=None,
                        choices=['window', 'segment'],
                        help="""Use streaming recognizer for inference.
                        `--batchsize` must be set to 0 to enable this mode""")
    parser.add_argument('--streaming-window', type=int, default=10,
                        help='Window size')
    parser.add_argument('--streaming-min-blank-dur', type=int, default=10,
                        help='Minimum blank duration threshold')
    parser.add_argument('--streaming-onset-margin', type=int, default=1,
                        help='Onset margin')
    parser.add_argument('--streaming-offset-margin', type=int, default=1,
                        help='Offset margin')
    # speech translation related
    parser.add_argument('--tgt-lang', default=False, type=str,
                        help='target language ID (e.g., <en>, <de>, <fr> etc.)')
    # Frontend related
    parser.add_argument('--with-category', type=strtobool, default=False,
                        help='Include category info in each minibatch')
    parser.add_argument('--wpe-iterations', type=int, default=1,
                        help='iterations for performing WPE')
    parser.add_argument('--use-beamformer', type=strtobool,
                        default=True, help='')
    parser.add_argument('--use-wpe-for-mix', type=strtobool, default=False,
                        help='Whether to separate mixture before WPE')
    parser.add_argument('--use-WPD-frontend', type=strtobool, default=False,
                        help='use WPD frontend instead of WPE + MVDR beamformer')
    parser.add_argument('--wpd-opt', type=float, default=1, choices=[1, 2, 3, 4, 5, 5.2, 5.3, 6],
#    parser.add_argument('--wpd-opt', type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
                        help='which WPD implementation to be used')
    parser.add_argument('--project', type=strtobool, default=False,
                        help='use a linear output layer after beamforming')
    parser.add_argument('--test-oracle', type=strtobool, default=False,
                        help='test the frontend with oracle IRMs')
    parser.add_argument('--decode-dir', type=str,
                        help='path to the decoding directory')
    parser.add_argument('--target-is-singlech', type=strtobool, default=False,
                        help='target clean speech is single channel')
    parser.add_argument('--use-random-model', type=strtobool, default=False,
                        help='use a randomly initialized model for evaluation')
    parser.add_argument('--test-wtaps', type=int, default=-1,
                        help='set number of filter taps (length) during testing, same as training by default')
    parser.add_argument('--test-num-mics', type=int, default=-1,
                        help='set number of microphones during testing, same as training by default')
    parser.add_argument('--load-from-mdl', default='', nargs='?',
                        help='Load model weights from another model')
    parser.add_argument('--target-is-mask', type=strtobool, default=False,
                        help='use IRM as the training target')
    parser.add_argument('--train-mask-only', type=strtobool, default=False,
                        help='only train the MaskEstimator()')
    return parser


def main(cmd_args):
    parser = get_parser()
    args = parser.parse_args(cmd_args)

    if args.ngpu == 0 and args.dtype == "float16":
        raise ValueError(f"--dtype {args.dtype} does not support the CPU backend.")

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(mn5k): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info('set random seed = %d' % args.seed)

    # train
    logging.info('backend = ' + args.backend)

    assert args.num_spkrs > 1
    if args.num_spkrs == 1:
        raise ValueError("single-speaker is not supported.")
    else:
        if args.backend == "pytorch":
            logging.warning('Multi-Speaker Speech Separation (evaluation)')
            from ss_train import evaluate
            #from ss_train import evaluate_v2 as evaluate
            evaluate(args)
        else:
            raise ValueError("Only pytorch is supported.")


if __name__ == '__main__':
    main(sys.argv[1:])
