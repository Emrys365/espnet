#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Automatic speech recognition model training script."""

import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import uuid

from distutils.version import LooseVersion

import configargparse
import numpy as np
import torch

from espnet.bin.asr_train import get_parser as get_parser_ori
from espnet.utils.cli_utils import strtobool
from espnet.utils.training.batchfy import BATCH_COUNT_CHOICES

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2")


# NOTE: you need this func to generate our sphinx doc
def get_parser(parser=None, required=True):
    """Get default arguments."""
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Train an automatic speech recognition (ASR) model on one CPU, "
            "one or multiple GPUs with DistributedDataParallel",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )

    # distributed data parallel config (for single-speaker case)
    parser.add_argument('--rank', default=0, type=int, help='rank of worker')
    parser.add_argument('--world-size', default=1, type=int, help='number of workers')
    # distributed data parallel config (for multi-speaker case)
    parser.add_argument(
        "--multiprocessing-distributed",
        type=strtobool,
        default=True,
        help="Distributed method is used when single-node mode.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Specify the port number of master"
        "Master is a host machine has RANK0 process.",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default=None,
        help="Specify the address s of master. "
        "Master is a host machine has RANK0 process.",
    )
    parser.add_argument(
        "--init-file-prefix",
        type=str,
        default=".dist_init_",
        help="The file name prefix for init_file, which is used for "
        "'Shared-file system initialization'. "
        "This option is used when --port is not specified",
    )
    parser.add_argument("--num-nodes", type=int, default=1, help="The number of nodes")
    # --ngpu means the number of GPUs per node in this mode
    return parser


def main(cmd_args):
    """Run the main training function."""
    parser = get_parser_ori()
    parser = get_parser(parser=parser)
    args, _ = parser.parse_known_args(cmd_args)
    if args.backend == "chainer" and args.train_dtype != "float32":
        raise NotImplementedError(
            f"chainer backend does not support --train-dtype {args.train_dtype}."
            "Use --dtype float32."
        )
    if args.ngpu == 0 and args.train_dtype in ("O0", "O1", "O2", "O3", "float16"):
        raise ValueError(
            f"--train-dtype {args.train_dtype} does not support the CPU backend."
        )

    from espnet.utils.dynamic_import import dynamic_import

    if args.model_module is None:
        model_module = "espnet.nets." + args.backend + "_backend.e2e_asr:E2E"
    else:
        model_module = args.model_module
    model_class = dynamic_import(model_module)
    model_class.add_arguments(parser)

    args = parser.parse_args(cmd_args)
    args.model_module = model_module
    if "chainer_backend" in args.model_module:
        args.backend = "chainer"
    if "pytorch_backend" in args.model_module:
        args.backend = "pytorch"

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format=f"[{os.uname()[1].split('.')[0]}"
            f"%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format=f"[{os.uname()[1].split('.')[0]}"
            f"%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # If --ngpu is not given,
    #   1. if CUDA_VISIBLE_DEVICES is set, all visible devices
    #   2. if nvidia-smi exists, use all devices
    #   3. else ngpu=0
    if args.ngpu is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            ngpu = len(cvd.split(","))
        else:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
            try:
                p = subprocess.run(
                    ["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                ngpu = 0
            else:
                ngpu = len(p.stderr.decode().split("\n")) - 1
    else:
        if is_torch_1_2_plus and args.ngpu != 1:
            logging.debug(
                "There are some bugs with multi-GPU processing in PyTorch 1.2+"
                + " (see https://github.com/pytorch/pytorch/issues/21108)"
            )
        ngpu = args.ngpu
    logging.warning(f"ngpu: {ngpu}")

    # display PYTHONPATH
    logging.warning("python path = " + os.environ.get("PYTHONPATH", "(None)"))
    logging.warning("pytorch = {}".format(torch.__version__))

    # set random seed
    logging.warning("random seed = %d" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dictionary for debug log
    if args.dict is not None:
        with open(args.dict, "rb") as f:
            dictionary = f.readlines()
        char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
        char_list.insert(0, "<blank>")
        char_list.append("<eos>")
        args.char_list = char_list
    else:
        args.char_list = None

    # train
    logging.info("backend = " + args.backend)

    if args.num_spkrs == 1:
        ##################################
        # distributed setting (for apex) #
        ##################################
        assert args.ngpu == 1, "In DDP Training mode, ngpu must be 1"
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir, exist_ok=True)
        sync_file = os.path.abspath(args.outdir) + '/synchronized'
        if args.rank == 0:
            if os.path.exists(sync_file):
                os.remove(sync_file)
        torch.distributed.init_process_group(
            backend = 'nccl',
            init_method = 'file://' + sync_file,
            world_size = args.world_size,
            rank = args.rank
        )
        torch.set_num_threads(2)
        # ------------------------------------------------------------

        if args.backend == "chainer":
            raise Exception('ddp training currently not support chainer mode')
            from espnet.asr.chainer_backend.asr import train

            train(args)
        elif args.backend == "pytorch":
            from espnet.asr.pytorch_backend.asr_ddp import train

            train(args)
        else:
            raise ValueError("Only chainer and pytorch are supported.")
    else:
        #############################################
        # distributed setting (ported from espnet2) #
        #############################################
        from espnet2.train.distributed_utils import resolve_distributed_mode

        args.dist_backend = "nccl"
        args.dist_world_size = None
        args.dist_rank = None
        args.local_rank = None
        args.dist_master_addr = None
        args.dist_master_port = None
        # "distributed" is decided using the other command args
        # (other related arguments in `args` will also be modified)
        resolve_distributed_mode(args)
        # ------------------------------------------------------------

        # FIXME(kamo): Support --model-module
        if args.backend == "pytorch":
            if args.mimo_with_ss_loss:
                from espnet.asr.pytorch_backend.asr_mix_with_ss_ddp import train
            else:
                from espnet.asr.pytorch_backend.asr_mix_ddp import train

            train(args)
        else:
            raise ValueError("Only pytorch is supported.")


if __name__ == "__main__":
    main(sys.argv[1:])
