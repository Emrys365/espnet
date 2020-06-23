#!/usr/bin/env python3

"""
This script is used for multi-speaker speech recognition.

Copyright 2017 Johns Hopkins University (Shinji Watanabe)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import json
import logging
import os

from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
from chainer.training.updater import StandardUpdater
from itertools import zip_longest as zip_longest
import math
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.nn.parallel import data_parallel

from espnet.asr.asr_mix_utils import add_results_to_json
from espnet.asr.asr_mix_utils import add_results_to_json_wer
from espnet.asr.asr_utils import adadelta_eps_decay

from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr import load_trained_model
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.e2e_asr import pad_list
#from espnet.nets.pytorch_backend.e2e_asr_mix import E2E
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

from espnet.nets.pytorch_backend.e2e_asr_mix_transformer import CTC_LOSS_THRESHOLD

import matplotlib
matplotlib.use('Agg')


class CustomUpdater(StandardUpdater):
    """Custom Updater for Pytorch.

    Args:
        model (torch.nn.Module): The model to update.
        grad_clip_threshold (float): The gradient clipping value to use.
        train_iter (chainer.dataset.Iterator): The training iterator.
        optimizer (torch.optim.optimizer): The training optimizer.

        converter (espnet.asr.pytorch_backend.asr.CustomConverter): Converter
            function to build input arrays. Each batch extracted by the main
            iterator and the ``device`` option are passed to this function.
            :func:`chainer.dataset.concat_examples` is used by default.

        device (torch.device): The device to use.
        ngpu (int): The number of gpus to use.
        use_apex (bool): The flag to use Apex in backprop.

    """

    def __init__(self, model, grad_clip_threshold, train_iter,
                 optimizer, converter, device, ngpu, grad_noise=False, accum_grad=1, use_apex=False):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.accum_grad = accum_grad
        self.forward_count = 0
        self.grad_noise = grad_noise
        self.iteration = 0
        self.invalid_samples = []
        self.use_apex = use_apex
        #----- added by wyz97
        self.use_multich_data = True

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main update routine of the CustomUpdater."""
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.next()
        # self.iteration += 1 # Increase may result in early report, which is done in other place automatically.
        x = self.converter(batch, self.device)

        # Compute the loss at this time step and accumulate it
        if self.ngpu == 0:
            loss = self.model(*x).mean() / self.accum_grad
        else:
            # apex does not support torch.nn.DataParallel
            loss = data_parallel(self.model, x, range(self.ngpu)).mean() / self.accum_grad
        if self.use_apex:
            from apex import amp
            # NOTE: for a compatibility with noam optimizer
            opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # gradient noise injection
        if self.grad_noise:
            from espnet.asr.asr_utils import add_gradient_noise
            add_gradient_noise(self.model, self.iteration, duration=100, eta=1.0, scale_factor=0.55)
        loss.detach()  # Truncate the graph

        # update parameters
        self.forward_count += 1
        if self.forward_count != self.accum_grad:
            return
        self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))

        if float(loss) >= CTC_LOSS_THRESHOLD or math.isnan(grad_norm):
            self.invalid_samples.append(self.iteration)
            print('bad sample at iteration {}.'.format(self.iteration))
            logging.warning('grad norm is nan. Do not update model.')
        # else:
        #     optimizer.step()
        optimizer.zero_grad()

    def update(self):
        self.update_core()
        # #iterations with accum_grad > 1
        # Ref.: https://github.com/espnet/espnet/issues/777
        #if self.forward_count == 0:
        self.iteration += 1


def filtering_train_json(train_json, sample_ids):
    """Filtering out the invalid samples from the original train_json.

    Args:
        train_json (dict): Dictionary of training data.
        sample_ids (list): List of ids of samples to be filtered out.

    Returns:
        new_train_json (dict): Filtered dictionary of training data.
    """
    new_train_json = train_json.copy()
    for sample in sample_ids:
        new_train_json.pop(sample)
        print("'{}' removed".format(sample))
    return new_train_json


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

    def __call__(self, batch, device):
        """Transform a batch and send it to a device.

        Args:
            batch (list(tuple(str, dict[str, dict[str, Any]]))): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor): Transformed batch.

        """
        # batch should be located in list
        assert len(batch) == 1
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


def load_pretrained_modules(model_path, target_model, match_keys, freeze_parms=False):
    src_model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    tgt_model_dict = target_model.state_dict()

    from collections import OrderedDict
    import re
    print('initialize: ', match_keys)
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

    return target_model


def pse_train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    # overridden some arguments to save time and memory cost
    args.batch_size = 1
    args.epochs = 1
    args.lsm_type = ''
    args.rnnlm = None
    args.ngpu = 1
    args.sortagrad = 0

    set_deterministic_pytorch(args)
#    torch.autograd.set_detect_anomaly(True)     # for debug

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][-1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][-1])
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

    # specify model architecture
    #model = E2E(idim, odim, args)
    model_class = dynamic_import(args.model_module)
    model = model_class(idim, odim, args)
    assert isinstance(model, ASRInterface)
    subsampling_factor = model.subsample[0]
    logging.warning('E2E model:\n{}'.format(model))

    # load pretrained model
    if args.init_frontend and args.init_asr:
        match_keys = r'^frontend\..*' #r'\.enc\..*' # r'^(?!.*enc_sd).*$'
        load_pretrained_modules(args.init_frontend, model, match_keys, freeze_parms=False)
#        match_keys = r'^(?!.*frontend).*'
        match_keys = r'(encoder|decoder|ctc)'
        load_pretrained_modules(args.init_asr, model, match_keys, freeze_parms=False)
#        torch_load(args.init_model_path, model)
        logging.info("Loading pretrained model " + args.init_frontend + " and " + args.init_asr)
    elif args.init_frontend:
        match_keys = r'^frontend\..*' #r'\.enc\..*' # r'^(?!.*enc_sd).*$'
        load_pretrained_modules(args.init_frontend, model, match_keys, freeze_parms=False)
#        torch_load(args.init_model_path, model)
        logging.info("Loading pretrained model " + args.init_frontend)
    elif args.init_asr:
        #match_keys = r'^(?!.*frontend).*' #r'\.enc\..*' # r'^(?!.*enc_sd).*$'
        #load_pretrained_modules(args.init_asr, model, match_keys, freeze_parms=False)
        match_keys = r'(encoder|decoder|ctc)'
        load_pretrained_modules(args.init_asr, model, match_keys, freeze_parms=True)
#        torch_load(args.init_model_path, model)
        logging.info("Loading pretrained model " + args.init_asr)

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch.load(args.rnnlm, rnnlm)
        model.rnnlm = rnnlm

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
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
            logging.info('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)

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
    converter = CustomConverter(subsampling_factor=subsampling_factor, dtype=dtype)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
#    with open(args.valid_json, 'rb') as f:
#        valid_json = json.load(f)['utts']

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=use_sortagrad,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          iaxis=0, oaxis=-1)
#    valid = make_batchset(valid_json, args.batch_size,
#                          args.maxlen_in, args.maxlen_out, args.minibatches,
#                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
#                          count=args.batch_count,
#                          batch_bins=args.batch_bins,
#                          batch_frames_in=args.batch_frames_in,
#                          batch_frames_out=args.batch_frames_out,
#                          batch_frames_inout=args.batch_frames_inout,
#                          iaxis=0, oaxis=-1)

    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    # load_cv = LoadInputsAndTargets(
    #     mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
    #     preprocess_args={'train': False}  # Switch the mode of preprocessing
    # )
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    if args.n_iter_processes > 0:
        train_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(train, load_tr),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20,
            shuffle=not use_sortagrad)
#            shuffle=False)
        # valid_iter = ToggleableShufflingMultiprocessIterator(
        #     TransformDataset(valid, load_cv),
        #     batch_size=1, repeat=False, shuffle=False,
        #     n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = ToggleableShufflingSerialIterator(
            TransformDataset(train, load_tr),
            batch_size=1, shuffle=not use_sortagrad)
#            batch_size=1, shuffle=False)
        # valid_iter = ToggleableShufflingSerialIterator(
        #     TransformDataset(valid, load_cv),
        #     batch_size=1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, train_iter, optimizer,
        converter, device, args.ngpu, args.grad_noise, args.accum_grad, use_apex=use_apex)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    if use_sortagrad:
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    # trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device, args.ngpu))

    # Save attention weight each epoch
    att_reporter = None

    # set 
    # if args.multich_epochs >= 0:
    #     t_epochs = list(range(args.multich_epochs, args.epochs))
    #     assert len(t_epochs) > 0
    #     updater.use_multich_data = False
    #     trainer.extend(schedule_multich_data(t_epochs))
    # else:
    updater.use_multich_data = True

    # Run the training
    trainer.run()
    # check_early_stop(trainer, args.epochs)

    # ids of the invalid samples
    sample_ids = [train[i][0][0] for i in updater.invalid_samples]
    # get filtered train_json without invalid samples
    new_train_json = filtering_train_json(train_json, sample_ids)
    jsonstring = json.dumps({'utts': new_train_json}, indent=4, ensure_ascii=False, sort_keys=True)
    with open(args.output_json_path, 'w') as f:
        f.write(jsonstring)
