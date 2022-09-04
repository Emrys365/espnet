from collections import defaultdict
import json
import logging
import os
import random
import re
import sys

import torch

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr import _recursive_to
from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.asr.pytorch_backend.asr_mix import CustomConverter
from espnet.bin.asr_train import get_parser
from espnet.nets.asr_interface import ASRInterface
from espnet.utils.cli_utils import strtobool
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet2.torch_utils.model_summary import model_summary


def parse_args(cmd_args):
    parser = get_parser(required=False)
    parser.add_argument("--model", type=str, required=True, help="Model file parameters to read")
    parser.add_argument("--model-conf", type=str, default=None, help="Model config file")
    # Resolve the frequency permutation problem via DOA estimation (for T-F masking based beamforming)
    parser.add_argument("--resolve-freq-perm", type=strtobool, default=False, help="Whether to resolve the frequency permutation problem via DOA estimation")
    parser.add_argument("--freq-perm-thres", type=float, default=180.0, help="Threshold used when resolving the frequency permutation problem via DOA estimation")
    parser.add_argument("--sensor-pos-json", type=str, default="", help="Path to the json file containing sensor position information for each sample")
    parser.add_argument("--fs", type=int, default=16000, help="Sampling rate of the data")
    args, _ = parser.parse_known_args(cmd_args)

    if args.backend == "chainer" and args.train_dtype != "float32":
        raise NotImplementedError(
        )
    if args.ngpu == 0 and args.train_dtype in ("O0", "O1", "O2", "O3", "float16"):
        raise ValueError(
            f"--train-dtype {args.train_dtype} does not support the CPU backend."
        )

    if args.model_module is None:
        model_module = "espnet.nets." + args.backend + "_backend.e2e_asr_mix_transformer:E2E"
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
            level=logging.warning,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")
    return args


def measure_gpu_max_memory_usage():
    assert torch.cuda.is_available()
    mem = torch.cuda.max_memory_allocated()
    return mem / 2**30


if __name__ == "__main__":
    cmd_args = sys.argv[1:]
    args = parse_args(cmd_args)
    assert args.batch_size == 1, args.batch_size
    random.seed(args.seed)

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

    set_deterministic_pytorch(args)
    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    if args.model_module.endswith('e2e_asr_mix_transformer_1ch:E2E'):
        args.test_nmics = 1

    if args.model_conf is not None:
        idim, odim, train_args = get_model_conf(args.model, args.model_conf)
        logging.warning('reading model parameters from ' + args.model)

        if hasattr(train_args, "model_module"):
            model_module = train_args.model_module
        else:
            model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"
        model_class = dynamic_import(model_module)
        model = model_class(idim, odim, train_args)
        torch_load(args.model, model)
    else:
        model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)

    # check the use of gpu
    assert args.ngpu == 1, args.ngpu
    # set torch device
    device = torch.device("cuda")
    logging.warning("device: {}".format(device))
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)

    # model summary
    if getattr(model, "frontend", None) is not None:
        msg = model_summary(model.frontend)
        msg = re.search(r"Model summary:\n.*", msg, flags=re.DOTALL).group()
        msg = "(Frontend) " + msg
        print(msg)
    msg = model_summary(model)
    msg = re.search(r"Model summary:\n.*", msg, flags=re.DOTALL).group()
    print(msg)

    if args.resolve_freq_perm:
        print("Resolving frequency permutation problem via DOA estimation")
        assert args.batchsize == 0, args.batchsize
        assert os.path.exists(args.sensor_pos_json), args.sensor_pos_json
        with open(args.sensor_pos_json, "r") as f:
            sensor_pos_info = json.load(f)

    subsampling_factor = model.subsample[0]
    # Setup a converter
    converter = CustomConverter(subsampling_factor=subsampling_factor, dtype=torch.float64)

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=args.lr, rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    elif args.opt == "noam_reducelronplateau":
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt_reducelronplateau

        optimizer = get_std_opt_reducelronplateau(
            model, args.adim, args.transformer_warmup_steps, args.transformer_lr
        )
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)


    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    js_keys = sorted(train_json.keys(), key=lambda k: -train_json[k]['input'][0]['shape'][0])

    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=False,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          iaxis=0, oaxis=-1)

    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        load_wav_ref=args.load_wav_ref,
        preprocess_args={'train': True},  # Switch the mode of preprocessing
        test_nmics=getattr(args, 'test_nmics', -1)
    )
    if getattr(args, 'test_nmics', -1) > 0:
        logging.warning('Using %d-channel data (randomly selected) for training' % args.test_nmics)

    model.train()
    # First run a random sample to warm up
    random_key = random.choice(js_keys)
    feat = load_tr([(random_key, train_json[random_key])])
    batch = converter([feat], device=device)
    xs_pad, ilens, ys_pad = _recursive_to(batch, device)
    loss = model(xs_pad, ilens, ys_pad)
    loss.backward()

    mems = defaultdict(list)
    for idx, name in enumerate(js_keys, 1):
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        if idx > 6:
            # There are 6 longest samples with a length of around 24s
            break

        batch = [(name, train_json[name])]
        feat = load_tr(batch)
        feat = (feat[0], list(feat[1]))
        if feat[0][0].ndim == 2:
            # skip single-channel data
            continue
        logging.warning('(%d/%d) training ' + name, idx, len(js_keys))
        logging.warning('speech size: {}; text sizes: {}'.format(feat[0][0].shape, [t.shape for t in feat[1][0]]))
        batch = converter([feat], device=device)
        xs_pad, ilens, ys_pad = _recursive_to(batch, device)
        mem_pre = measure_gpu_max_memory_usage()
        logging.warning("mem_pre: %f GiB" % mem_pre)
        loss = model(xs_pad, ilens, ys_pad)
        mem_forward = measure_gpu_max_memory_usage()
        logging.warning("mem_forward: %f GiB" % mem_forward)
        loss.backward()
        mem_backward = measure_gpu_max_memory_usage()
        logging.warning("mem_backward: %f GiB" % mem_backward)
        mems["pre"].append(mem_pre)
        mems["forward"].append(mem_forward)
        mems["backward"].append(mem_backward)
        print()

    print({name: sum(l) / len(l) for name, l in mems.items()})


"""Usage example:

# MVDR_Souden, T-F mask (2ch)
CUDA_VISIBLE_DEVICES=0 python measure_gpu_memory_usage.py \
    --config conf/tuning/train_multispkr_trans_wyz97_padertorch_mvdr.yaml \
    --model exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_2021_05_22/results/model.acc.best \
    --ngpu 1 \
    --backend pytorch \
    --debugmode 1 \
    --dict data/lang_1char/tr_units.txt \
    --minibatches 0 \
    --verbose 0 \
    --train-json data/train_si284_singlespkr/data.json \
    --preprocess-conf conf/preprocess.yaml \
    --num-spkrs 2 \
    --use-WPD-frontend False \
    --load-wav-ref False \
    --seed 1 \
    --test-nmics 2 \
    --use-padertorch-frontend True \
    --batch-size 1 \
    --ctc_type builtin \
    --fbank-fs 8000


# MVDR_Souden, T-F mask (6ch, TBPTT)
CUDA_VISIBLE_DEVICES=0 python measure_gpu_memory_usage.py \
    --config conf/tuning/train_multispkr_trans_wyz97_padertorch_mvdr_tbptt.yaml \
    --model exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_tbptt_preprocess_uttcmvn_6ch_5taps_2021_12_22/results/model.acc.best \
    --ngpu 1 \
    --backend pytorch \
    --debugmode 1 \
    --dict data/lang_1char/tr_units.txt \
    --minibatches 0 \
    --verbose 0 \
    --train-json data/train_si284_singlespkr/data.json \
    --preprocess-conf conf/preprocess.yaml \
    --num-spkrs 2 \
    --use-WPD-frontend False \
    --load-wav-ref False \
    --seed 1 \
    --use-padertorch-frontend True \
    --batch-size 1 \
    --ctc_type builtin \
    --fbank-fs 8000


# MVDR_Souden, VAD-like mask (6ch, TBPTT)
CUDA_VISIBLE_DEVICES=0 python measure_gpu_memory_usage.py \
    --config conf/tuning/train_multispkr_trans_wyz97_padertorch_mvdr_tbptt_vad.yaml \
    --model exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_tbptt_vad_preprocess_uttcmvn_6ch_5taps_vad_mask_2022_01_31/results/model.acc.best \
    --ngpu 1 \
    --backend pytorch \
    --debugmode 1 \
    --dict data/lang_1char/tr_units.txt \
    --minibatches 0 \
    --verbose 0 \
    --train-json data/train_si284_singlespkr/data.json \
    --preprocess-conf conf/preprocess.yaml \
    --num-spkrs 2 \
    --use-WPD-frontend False \
    --load-wav-ref False \
    --seed 1 \
    --use-padertorch-frontend True \
    --use-vad-mask True \
    --batch-size 1 \
    --ctc_type builtin \
    --fbank-fs 8000
"""
