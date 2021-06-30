#!/usr/bin/env python3

"""
This script is used for multi-speaker speech recognition.

Copyright 2017 Johns Hopkins University (Shinji Watanabe)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import argparse
from distutils.version import LooseVersion
import json
import logging
import os

from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
from itertools import zip_longest as zip_longest
import numpy as np
from tensorboardX import SummaryWriter
import torch

from espnet.asr.asr_mix_utils import add_results_to_json
from espnet.asr.asr_mix_utils import add_results_to_json_wer
from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import adam_lr_decay

from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr import CustomEvaluator
from espnet.asr.pytorch_backend.asr import CustomUpdater
from espnet.asr.pytorch_backend.asr import load_trained_model
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.e2e_asr import pad_list
#from espnet.nets.pytorch_backend.e2e_asr_mix import E2E
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ShufflingEnabler
#from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
#from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.distributed_utils import free_port
from espnet2.train.distributed_utils import get_master_port
from espnet2.train.distributed_utils import get_node_rank
from espnet2.train.distributed_utils import get_num_nodes
from espnet2.utils.build_dataclass import build_dataclass

import matplotlib
matplotlib.use('Agg')

if LooseVersion(torch.__version__) >= LooseVersion("1.5.0"):
    from torch.multiprocessing.spawn import ProcessContext
else:
    from torch.multiprocessing.spawn import SpawnContext as ProcessContext


class CustomConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsampling_factor=1, dtype=torch.float32):
        """Initialize the converter."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list(tuple(str, dict[str, dict[str, Any]]))): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor): Transformed batch.

        """
        # batch should be located in list
        assert len(batch) == 1
        #print([b.shape if getattr(b, 'shape', None) else len(list(b)) for b in batch[0]], flush=True)
        xs, ys = batch[0]
        # Convert zip object to list in python 3.x
        ys = list(ys)

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        if xs[0].dtype.kind == 'c':
            xs_pad_real = pad_list(
                [torch.from_numpy(x.real).float() for x in xs], 0).to(device, dtype=self.dtype)
            xs_pad_imag = pad_list(
                [torch.from_numpy(x.imag).float() for x in xs], 0).to(device, dtype=self.dtype)
            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it to E2E here
            # because torch.nn.DataParallel can't handle it.
            xs_pad = {'real': xs_pad_real, 'imag': xs_pad_imag}
        else:
            xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device, dtype=self.dtype)

        ilens = torch.from_numpy(ilens).to(device)
        # TODO(Xuankai): try to make this neat
        if not isinstance(ys[0], np.ndarray):
            ys_pad = [torch.from_numpy(y[0]).long() for y in ys] + [torch.from_numpy(y[1]).long() for y in ys]
            ys_pad = pad_list(ys_pad, self.ignore_id)
            ys_pad = ys_pad.view(2, -1, ys_pad.size(1)).transpose(0, 1).to(device)  # (num_spkrs, B, Tmax)
        else:
            ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], self.ignore_id).to(device)

        return xs_pad, ilens, ys_pad


def schedule_multich_data(epoch_list):
    """Determine when to introduce the multichannel data, scheduled on epoch_list.

    Example usage:
    trainer.extend(schedule_multich_data(3))
    """
    if not isinstance(epoch_list, list):
        assert isinstance(epoch_list, float) or isinstance(epoch_list, int)
        epoch_list = [epoch_list, ]

    trigger = training.triggers.ManualScheduleTrigger(epoch_list, 'epoch')
    # count = 0

    @training.extension.make_extension(trigger=trigger)
    def set_value(trainer):
        # nonlocal count
        updater = trainer.updater
        print('[epoch {}] updater.use_multich_data: {} -> True'.format(
              updater.epoch_detail, updater.use_multich_data))
        setattr(updater, 'use_multich_data', True)
        # count += 1

    return set_value


def load_pretrained_modules(model_path, target_model, match_keys, freeze_parms=False, map_location=None):
    if map_location is None:
        src_model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        src_model_dict = torch.load(model_path, map_location=map_location)
    tgt_model_dict = target_model.state_dict()

    from collections import OrderedDict
    import re
    print('initialize: ', match_keys, flush=True)
    filtered_keys = filter(lambda x: re.search(match_keys, x[0]), src_model_dict.items())
    filtered_dict = OrderedDict()
    for key, v in filtered_keys:
        filtered_dict[key] = v

    tgt_model_dict.update(filtered_dict)
    target_model.load_state_dict(tgt_model_dict)

    if freeze_parms:
        for name, param in target_model.named_parameters():
            if name in filtered_dict:
                param.requires_grad = False
                print(f'{name}: freezed', flush=True)
            else:
                print(f'{name}: requires_grad={param.requires_grad}', flush=True)

    return target_model


def init_wpd_model_from_mvdr_wpe(model_path, target_model, freeze_parms=False, map_location=None):
    if map_location is None:
        src_model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        src_model_dict = torch.load(model_path, map_location=map_location)
    tgt_model_dict = target_model.state_dict()

    from collections import OrderedDict
    import re
    filtered_dict = OrderedDict()
    for key, v in src_model_dict.items():
        if 'frontend.beamformer' in key:
            key2 = re.sub(r'frontend.beamformer', 'frontend', key)
            if key2 in tgt_model_dict:
                filtered_dict[key2] = v
        elif key in tgt_model_dict:
            filtered_dict[key] = v

    tgt_model_dict.update(filtered_dict)
    target_model.load_state_dict(tgt_model_dict)

    if freeze_parms:
        for name, param in target_model.named_parameters():
            if name in filtered_dict and re.search(r'(encoder|decoder|ctc)', name):
                    param.requires_grad = False

    return target_model


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    if not args.distributed or not args.multiprocessing_distributed:
        train_main_worker(args)
    else:
        assert args.ngpu > 1, args.ngpu
        # Multi-processing distributed mode: e.g. 2node-4process-4GPU
        # |   Host1     |    Host2    |
        # |   Process1  |   Process2  |  <= Spawn processes
        # |Child1|Child2|Child1|Child2|
        # |GPU1  |GPU2  |GPU1  |GPU2  |

        # See also the following usage of --multiprocessing-distributed:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        num_nodes = get_num_nodes(args.dist_world_size, args.dist_launcher)
        if num_nodes == 1:
            args.dist_master_addr = "localhost"
            args.dist_rank = 0
            # Single node distributed training with multi-GPUs
            if (
                args.dist_init_method == "env://"
                and get_master_port(args.dist_master_port) is None
            ):
                # Get the unused port
                args.dist_master_port = free_port()

        # Assume that nodes use same number of GPUs each other
        args.dist_world_size = args.ngpu * num_nodes
        node_rank = get_node_rank(args.dist_rank, args.dist_launcher)

        # The following block is copied from:
        # https://github.com/pytorch/pytorch/blob/master/torch/multiprocessing/spawn.py
        error_queues = []
        processes = []
        mp = torch.multiprocessing.get_context("spawn")
        for i in range(args.ngpu):
            # Copy args
            local_args = argparse.Namespace(**vars(args))

            local_args.local_rank = i
            local_args.dist_rank = args.ngpu * node_rank + i
            local_args.ngpu = 1

            process = mp.Process(
                target=train_main_worker,
                args=(local_args,),
                daemon=False,
            )
            process.start()
            processes.append(process)
            error_queues.append(mp.SimpleQueue())
        # Loop on join until it returns True or raises an exception.
        while not ProcessContext(processes, error_queues).join():
            pass


def train_main_worker(args):
    """Training process with the given args.

    Args:
        args (namespace): The program arguments.

    """
    # 0. Init distributed process
    distributed_option = build_dataclass(DistributedOption, args)
    # Setting distributed_option.dist_rank, etc.
    distributed_option.init_options()

    # Remove all handlers associated with the root logger object
    # before we reset the basicConfig
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if not distributed_option.distributed or distributed_option.dist_rank == 0:
        if not distributed_option.distributed:
            _rank = ""
        else:
            _rank = (
                f":{distributed_option.dist_rank}/"
                f"{distributed_option.dist_world_size}"
            )

        # NOTE(kamo):
        # logging.basicConfig() is invoked in train_main_worker() instead of main()
        # because it can be invoked only once in a process.
        # FIXME(kamo): Should we use logging.getLogger()?
        log_level = logging.INFO if args.verbose > 0 else logging.WARN
        logging.basicConfig(
            level=log_level,
            format=f"[{os.uname()[1].split('.')[0]}{_rank}]"
            f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        # Suppress logging if RANK != 0
        logging.basicConfig(
            level="ERROR",
            format=f"[{os.uname()[1].split('.')[0]}"
            f":{distributed_option.dist_rank}/{distributed_option.dist_world_size}]"
            f" %(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    # Invoking torch.distributed.init_process_group
    distributed_option.init_torch_distributed()

    set_deterministic_pytorch(args)
#    torch.autograd.set_detect_anomaly(True)     # for debug

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utt0 = next(iter(valid_json.keys()))
    idim = int(valid_json[utt0]['input'][0]['shape'][-1])
    odim = int(valid_json[utt0]['output'][0]['shape'][-1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    if args.model_module.endswith('e2e_asr_mix_transformer_1ch:E2E'):
        args.test_nmics = 1

    # specify model architecture
    #model = E2E(idim, odim, args)
    model_class = dynamic_import(args.model_module)
    model = model_class(idim, odim, args)
    assert isinstance(model, ASRInterface)
    subsampling_factor = model.subsample[0]
    logging.warning('E2E model:\n{}'.format(model))
    logging.warning(
        " Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )

    # load pretrained model
    if args.init_from_mdl:
        init_wpd_model_from_mvdr_wpe(
            args.init_from_mdl, model, freeze_parms=True,
            map_location=f"cuda:{torch.cuda.current_device()}" if args.ngpu > 0 else "cpu",
        )
        logging.info("Loading pretrained model " + args.init_from_mdl)

    elif args.init_frontend and args.init_asr:
        match_keys = r'^frontend\..*' #r'\.enc\..*' # r'^(?!.*enc_sd).*$'
        #load_pretrained_modules(args.init_frontend, model, match_keys, freeze_parms=True)
        load_pretrained_modules(
            args.init_frontend, model, match_keys, freeze_parms=False,
            map_location=f"cuda:{torch.cuda.current_device()}" if args.ngpu > 0 else "cpu"
        )
        match_keys = r'(encoder|decoder|ctc)'
        #match_keys = r'^(?!.*frontend).*'
#        load_pretrained_modules(args.init_asr, model, match_keys, freeze_parms=True)
        load_pretrained_modules(
            args.init_asr, model, match_keys, freeze_parms=False,
            map_location=f"cuda:{torch.cuda.current_device()}" if args.ngpu > 0 else "cpu"
        )
#        torch_load(args.init_model_path, model)
        logging.info("Loading pretrained model " + args.init_frontend + " and " + args.init_asr)

    elif args.init_frontend:
        match_keys = r'^frontend\..*' #r'\.enc\..*' # r'^(?!.*enc_sd).*$'
        load_pretrained_modules(
            args.init_frontend, model, match_keys, freeze_parms=False,
            map_location=f"cuda:{torch.cuda.current_device()}" if args.ngpu > 0 else "cpu"
        )
#        torch_load(args.init_model_path, model)
        logging.info("Loading pretrained model " + args.init_frontend)

    elif args.init_asr:
        #match_keys = r'^(?!.*frontend).*' #r'\.enc\..*' # r'^(?!.*enc_sd).*$'
        #load_pretrained_modules(args.init_asr, model, match_keys, freeze_parms=False)
        match_keys = r'(encoder|decoder|ctc)'
        #load_pretrained_modules(args.init_asr, model, match_keys, freeze_parms=True)
        load_pretrained_modules(
            args.init_asr, model, match_keys, freeze_parms=False,
            map_location=f"cuda:{torch.cuda.current_device()}" if args.ngpu > 0 else "cpu"
        )
#        torch_load(args.init_model_path, model)
        logging.info("Loading pretrained model " + args.init_asr)

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch.load(
            args.rnnlm, rnnlm,
            map_location=f"cuda:{torch.cuda.current_device()}" if args.ngpu > 0 else "cpu",
        )
        model.rnnlm = rnnlm

    # write model config
    model_conf = args.outdir + '/model.json'
    if not distributed_option.distributed or distributed_option.dist_rank == 0:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        with open(model_conf, 'wb') as f:
            logging.info('writing a model config file to ' + model_conf)
            f.write(json.dumps((idim, odim, vars(args)),
                            indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
        for key in sorted(vars(args).keys()):
            logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        if args.batch_size != 0:
            logging.warning('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    logging.warning("device: {}".format(device))
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    # train frontend on CPU to make it more stable

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=args.lr, rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # setup apex.amp
    if args.train_dtype in ("O0", "O1", "O2", "O3"):
        try:
            from apex import amp
        except ImportError as e:
            logging.error(f"You need to install apex for --train-dtype {args.train_dtype}. "
                          "See https://github.com/NVIDIA/apex#linux")
            raise e
        if args.opt == 'noam':
            model, optimizer.optimizer = amp.initialize(model, optimizer.optimizer, opt_level=args.train_dtype)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.train_dtype)
        use_apex = True
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
#    converter = CustomConverter(subsampling_factor=subsampling_factor, dtype=dtype)
    converter = CustomConverter(subsampling_factor=subsampling_factor, dtype=torch.float64)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(
        train_json, args.batch_size,
        args.maxlen_in, args.maxlen_out, args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0, oaxis=-1,
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )
    # note(yzl23): to avoid missing data in dev set (distributed into different workers), all workers compute whole devset
    # besides, compute all data makes statistics in devset the same over all workers, this property is helpful for lr scheduler
    valid = make_batchset(
        valid_json, args.batch_size,
        args.maxlen_in, args.maxlen_out, args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0, oaxis=-1,
    )

    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        load_wav_ref=args.load_wav_ref,
        preprocess_args={'train': True},  # Switch the mode of preprocessing
        test_nmics=getattr(args, 'test_nmics', -1)
    )
    load_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        load_wav_ref=args.load_wav_ref,
        preprocess_args={'train': False},  # Switch the mode of preprocessing
        test_nmics=getattr(args, 'test_nmics', -1)
    )
    if getattr(args, 'test_nmics', -1) > 0:
        logging.warning('Using %d-channel data (randomly selected) for training' % args.test_nmics)

    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    train_iter = ChainerDataLoader(
        dataset=TransformDataset(train, lambda data: converter([load_tr(data)])),
        batch_size=1,
        num_workers=args.n_iter_processes,
        shuffle=not use_sortagrad,
        collate_fn=lambda x: x[0],
    )
    valid_iter = ChainerDataLoader(
        dataset=TransformDataset(valid, lambda data: converter([load_cv(data)])),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
#        num_workers=args.n_iter_processes,
    )

    # Set up a trainer
    updater = CustomUpdater(
        model,
        args.grad_clip,
        {"main": train_iter},
        optimizer,
        device,
        args.ngpu,
        args.grad_noise,
        args.accum_grad,
        use_apex=use_apex,
    )

    if args.update_interval_iters > 0:
        trainer = training.Trainer(updater, (args.update_interval_iters, "iteration"), out=args.outdir)
    else:
        trainer = training.Trainer(updater, (args.epochs, "epoch"), out=args.outdir)

    if use_sortagrad:
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    if args.save_interval_iters > 0:
        trainer.extend(
            CustomEvaluator(model, {"main": valid_iter}, reporter, device, args.ngpu),
            trigger=(args.save_interval_iters, "iteration"),
        )
    else:
        trainer.extend(
            CustomEvaluator(model, {"main": valid_iter}, reporter, device, args.ngpu)
        )

    # Save attention weight each epoch
    if (not distributed_option.distributed or distributed_option.dist_rank == 0) and \
        args.num_save_attention > 0 and args.mtlalpha != 1.0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
            plot_class = model.module.attention_plot_class
        else:
            att_vis_fn = model.calculate_all_attentions
            plot_class = model.attention_plot_class
        att_reporter = plot_class(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, transform=load_cv, device=device)
        trainer.extend(att_reporter, trigger=(1, 'epoch'))
    else:
        att_reporter = None

    # Make a plot for training and validation values
    if not distributed_option.distributed or distributed_option.dist_rank == 0:
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                            'main/loss_ctc', 'validation/main/loss_ctc',
                                            'main/loss_att', 'validation/main/loss_att'],
                                            'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                            'epoch', file_name='loss_main.png'))
        trainer.extend(extensions.PlotReport(['main/loss_ctc', 'validation/main/loss_ctc'],
                                            'epoch', file_name='loss_ctc.png'))
        trainer.extend(extensions.PlotReport(['main/loss_att', 'validation/main/loss_att'],
                                            'epoch', file_name='loss_att.png'))
        trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                            'epoch', file_name='acc.png'))
        trainer.extend(extensions.PlotReport(['main/cer_ctc', 'validation/main/cer_ctc'],
                                            'epoch', file_name='cer.png'))

        # Save best models
        trainer.extend(snapshot_object(model, 'model.loss.best'),
                    trigger=training.triggers.MinValueTrigger('validation/main/loss'))
        if mtl_mode != 'ctc':
            trainer.extend(snapshot_object(model, 'model.acc.best'),
                        trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

        # save snapshot which contains model and optimizer states
        if args.save_interval_iters > 0:
            trainer.extend(
                torch_snapshot(filename="snapshot.iter.{.updater.iteration}"),
                trigger=(args.save_interval_iters, "iteration"),
            )
        else:
            trainer.extend(torch_snapshot(), trigger=(1, "epoch"))


    # Frontend require_grad after 1 epochs
    def resume_require_grad():
        @training.extension.make_extension(trigger=(1, 'epoch'), priority=-100)
        def set_require_grad(trainer):
            for parm in trainer.updater.model.parameters():
                parm.requires_grad = True
        return set_require_grad
    if args.init_from_mdl or args.init_asr:
        trainer.extend(resume_require_grad(), trigger=(1, 'epoch'))

    # set 
    if args.multich_epochs >= 0:
        t_epochs = list(range(args.multich_epochs, args.epochs))
        assert len(t_epochs) > 0
        updater.use_multich_data = False
        trainer.extend(schedule_multich_data(t_epochs))
    else:
        updater.use_multich_data = True

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        raise Exception('adadelta optimizer is not supported in DDP training mode')
        if args.criterion == 'acc' and mtl_mode != 'ctc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
    elif args.opt == "adam":
        if args.criterion == "acc" and mtl_mode != "ctc":
            trainer.extend(
                adam_lr_decay(0.5),
                trigger=CompareValueTrigger(
                    "validation/main/acc",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )
        elif args.criterion == "loss":
            trainer.extend(
                adam_lr_decay(0.5),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value > current_value,
                ),
            )
        logging.warning("optimize with adam lr decay 0.5")

    # Write a log of evaluation statistics for each epoch
    if not distributed_option.distributed or distributed_option.dist_rank == 0:
        trainer.extend(extensions.LogReport(trigger=(args.report_interval_iters, 'iteration')))
        report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                        'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                        'main/acc', 'validation/main/acc', 'main/cer_ctc', 'validation/main/cer_ctc',
                        'elapsed_time']
        if args.opt == 'adadelta':
            trainer.extend(extensions.observe_value(
                'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
                trigger=(args.report_interval_iters, 'iteration'))
            report_keys.append('eps')
        if args.report_cer:
            report_keys.append('validation/main/cer')
        if args.report_wer:
            report_keys.append('validation/main/wer')
        trainer.extend(extensions.PrintReport(
            report_keys), trigger=(args.report_interval_iters, 'iteration'))

        trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))

        if args.tensorboard_dir is not None and args.tensorboard_dir != "":
            trainer.extend(TensorboardLogger(SummaryWriter(args.tensorboard_dir), att_reporter),
                            trigger=(args.report_interval_iters, "iteration"))
    # Run the training
    set_early_stop(trainer, args)
    trainer.run()
    check_early_stop(trainer, args.epochs)


def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    if args.model_conf is not None:
        idim, odim, train_args = get_model_conf(args.model, args.model_conf)
        logging.info('reading model parameters from ' + args.model)

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
    model.recog_args = args
    if getattr(args, 'test_btaps', -1) > 0:
        if hasattr(model.frontend, 'taps'):
            model.frontend.taps = args.test_btaps
            logging.warning('setting taps to {}'.format(model.frontend.taps))
        if hasattr(model.frontend, 'btaps'):
            model.frontend.btaps = args.test_btaps
            logging.warning('setting btaps to {}'.format(model.frontend.btaps))
        if hasattr(model.frontend, 'wpe') and hasattr(model.frontend.wpe, 'taps'):
            model.frontend.wpe.taps = args.test_btaps
            logging.warning('setting wpe.taps to {}'.format(model.frontend.wpe.taps))
        if hasattr(model.frontend, 'beamformer') and hasattr(model.frontend.beamformer, 'btaps'):
            model.frontend.beamformer.btaps = args.test_btaps
            logging.warning('setting beamformer.btaps to {}'.format(model.frontend.beamformer.btaps))

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError("use '--api v2' option to decode with non-default language model")
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(word_dict),
                rnnlm_args.layer,
                rnnlm_args.unit,
                getattr(rnnlm_args, "embed_unit", None),  # for backward compatibility
            )
        )
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='asr', load_output=True, sort_in_input_length=False,
        load_wav_ref=getattr(train_args, 'load_wav_ref', True),
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf,
        preprocess_args={'train': False},
        test_nmics=getattr(args, 'test_nmics', -1)
    )
#        test_nmics=getattr(args, 'test_nmics', -1))

    if args.batchsize == 0:
        with torch.no_grad():
            for idx, name in enumerate(js.keys(), 1):
                logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
                batch = [(name, js[name])]
                feat = load_inputs_and_targets(batch)[0][0]
                nbest_hyps = model.recognize(feat, args, train_args.char_list, rnnlm)
                new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    else:
        def grouper(n, iterable, fillvalue=None):
            kargs = [iter(iterable)] * n
            return zip_longest(*kargs, fillvalue=fillvalue)

        # sort data if batchsize > 1
        keys = list(js.keys())
        if args.batchsize > 1:
            feat_lens = [js[key]['input'][0]['shape'][0] for key in keys]
            sorted_index = sorted(range(len(feat_lens)), key=lambda i: -feat_lens[i])
            keys = [keys[i] for i in sorted_index]

        setattr(args, "space", train_args.sym_space)
        setattr(args, "blank", train_args.sym_blank)
        errors = dict()
        wers, cers = [], []
        with torch.no_grad():
            for names in grouper(args.batchsize, keys, None):
                names = [name for name in names if name]
                batch = [(name, js[name]) for name in names]
                feats, labels = load_inputs_and_targets(batch)
#                wer, cer = model.calculate_error_batch(feats, list(labels), args, train_args.char_list, rnnlm=rnnlm)
#                # only report mean wer and cer for each batch
#                for name in names:
#                    errors[name] = {'cer': cer, 'wer': wer}
#                wers.append(wer)
#                cers.append(cer)
#
#        mean_wer = sum(wers) / float(len(wers))
#        mean_cer = sum(cers) / float(len(cers))
#
#        with open(args.result_label, 'wb') as f:
#            f.write('wer: {}\ncer: {}\n'.format(mean_wer, mean_cer).encode('utf_8'))
#            f.write(json.dumps({'utts': errors}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
#        return

                nbest_hyps = model.recognize_batch(feats, args, train_args.char_list, rnnlm=rnnlm)

                for i, name in enumerate(names):
                    nbest_hyp = [hyp[i] for hyp in nbest_hyps]
                    new_js[name] = add_results_to_json(js[name], nbest_hyp, train_args.char_list)
                    # new_js[name] = add_results_to_json_wer(js[name], nbest_hyp, list(labels), train_args.char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
