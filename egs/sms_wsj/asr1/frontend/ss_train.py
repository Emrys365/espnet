#!/usr/bin/env python3

"""
This script is used for multi-speaker speech enhancecment and separation.

Copyright 2017 Johns Hopkins University (Shinji Watanabe)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import io
import json
import logging
import os
import yaml

from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
import numpy as np
# from tensorboardX import SummaryWriter
import torch

from espnet.asr.asr_utils import adadelta_eps_decay

from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
#from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from separation import CustomConverter
from separation import CustomEvaluator
from separation import CustomUpdater

from batchfy import make_batchset
from data_io import LoadInputsAndTargets
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

import matplotlib
matplotlib.use('Agg')


def load_trained_model(model_path):
    """Load the trained model for recognition.

    Args:
        model_path(str): Path to model.***.best

    """
    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), 'model.json'))

    model = E2E(idim, odim, train_args)
    torch_load(model_path, model)

    return model, train_args


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utt0 = next(iter(valid_json))
    idim = int(valid_json[utt0]['input'][0]['shape'][-1])
    # odim = idim
    odim = 52
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # if args.target_is_mask and args.train_mask_only:
    #     from ss_mask_submodule import E2E
    # else:
    from ss_model import E2E

    # specify model architecture
    # global E2E
    # E2E = dynamic_import(args.model_module)
    model = E2E(idim, odim, args)
    del model.feature_transform
    del model.encoder
    del model.decoder
    del model.ctc
    del model.criterion
    # model = E2E(args, idim)
    assert isinstance(model, ASRInterface)
    subsampling_factor = model.subsample[0]
    logging.warning('E2E model:\n{}'.format(model))

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
    logging.warning("device: {}".format(device))
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
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode='min', factor=0.5, patience=1
    #)
    #scheduler = torch.optim.lr_scheduler.StepLR(
    #    optimizer, 2, gamma=0.98
    #)
    scheduler = None

    # Setup a converter
    converter = CustomConverter(subsampling_factor=subsampling_factor, dtype=dtype)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=use_sortagrad)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1)

    if args.load_input_lengths:
        logging.warning("Loading original input waveform lengths in each batch")
    load_tr = LoadInputsAndTargets(
        load_input_lengths=args.load_input_lengths, load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True},  # Switch the mode of preprocessing
        target_is_mask=args.target_is_mask,
        target_is_singlech=args.target_is_singlech,
        test_num_mics=args.test_nmics
    )
    load_cv = LoadInputsAndTargets(
        load_input_lengths=args.load_input_lengths, load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False},  # Switch the mode of preprocessing
        target_is_mask=args.target_is_mask,
        target_is_singlech=args.target_is_singlech,
        test_num_mics=args.test_nmics
    )
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    if args.n_iter_processes > 0:
        train_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(train, load_tr),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20,
            shuffle=not use_sortagrad)
#            shuffle=False)
        valid_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = ToggleableShufflingSerialIterator(
            TransformDataset(train, load_tr),
            batch_size=1, shuffle=not use_sortagrad)
#            batch_size=1, shuffle=False)
        valid_iter = ToggleableShufflingSerialIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(
        model, args.grad_clip, train_iter, optimizer, scheduler,
        converter, device, args.ngpu, args.grad_noise, args.accum_grad, use_apex=False)
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
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device, updater, args.ngpu))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                         'epoch', file_name='loss.png'))
    # trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
    #                                      'epoch', file_name='acc.png'))

    # Save best models
    trainer.extend(snapshot_object(model, 'model.loss.best'),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion != 'loss':
            raise ValueError('Unsupported criterion: ' + args.criterion)
        else:
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

#    # learning rate scheduler
#    def lr_drop(trainer):
#        # lower the learning rate every 2 epochs by multiplying 0.98 with the current learning rate.
#        for param_group in trainer.updater.get_optimizer('main').param_groups:
#            param_group['lr'] = param_group['lr'] * 0.98
#    trainer.extend(lr_drop, trigger=(2, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(args.report_interval_iters, 'iteration')))
    # --- observe_lr ---
    #trainer.extend(extensions.observe_lr())

    report_keys = ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(args.report_interval_iters, 'iteration'))
        report_keys.append('eps')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(args.report_interval_iters, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    set_early_stop(trainer, args)

    # if args.tensorboard_dir is not None and args.tensorboard_dir != "":
    #     trainer.extend(TensorboardLogger(SummaryWriter(args.tensorboard_dir), att_reporter),
    #                    trigger=(args.report_interval_iters, "iteration"))

    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


def load_from_asr_model(model_path, target_model, freeze_parms=False):
    src_model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
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
            if name in filtered_dict:
                param.requires_grad = False

    return target_model


def load_mask_estimator(model_path, target_model, freeze_parms=False):
    src_model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    tgt_model_dict = target_model.state_dict()
    logging.warning('loading mask model:\n{}'.format(src_model_dict.keys()))

    from collections import OrderedDict
    import re
    filtered_dict = OrderedDict()
    for key in tgt_model_dict.keys():
        if 'frontend.beamformer.mask' in key:
            key2 = re.sub(r'frontend.beamformer', 'frontend', key)
            filtered_dict[key] = src_model_dict[key2]
        elif 'frontend.mask' in key:
            filtered_dict[key] = src_model_dict[key]

    tgt_model_dict.update(filtered_dict)
    target_model.load_state_dict(tgt_model_dict)

    if freeze_parms:
        for name, param in target_model.named_parameters():
            if name in filtered_dict:
                param.requires_grad = False

    return target_model


def evaluate_v2(args):
    """Test with the given args.
        Using pb_bss_eval

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)

    idim, odim, train_args = get_model_conf(args.model, getattr(args, 'model_conf', None))
    target_is_mask = getattr(train_args, "target_is_mask", False)
    train_mask_only = getattr(train_args, "train_mask_only", False)
    # global E2E
    # E2E = dynamic_import(train_args.model_module)
    # if target_is_mask and train_mask_only:
    #     E2E = dynamic_import('ss_mask_submodule:E2E')
    #     #from ss_mask_submodule import E2E
    # else:
    #     E2E = dynamic_import('ss_model:E2E')
    from ss_model import E2E

    if args.load_from_mdl:
        model = E2E(idim, odim, train_args)
        load_from_asr_model(args.load_from_mdl, model, freeze_parms=True)
        logging.info("Loading pretrained model " + args.load_from_mdl)
    else:
        model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.recog_args = args
    if args.test_wtaps != -1:
        if hasattr(model.frontend, 'btaps'):
            model.frontend.btaps = args.test_wtaps
        elif hasattr(model.frontend, 'taps'):
            model.frontend.taps = args.test_wtaps
        if hasattr(model.frontend, 'wpe'):
            model.frontend.wpe.taps = args.test_wtaps

    #model.frontend.ref

    conf = train_args.preprocess_conf if args.preprocess_conf is None else args.preprocess_conf

    # get content of preprocess_conf
    with io.open(conf, encoding='utf-8') as f:
        conf_dict = yaml.safe_load(f)
        assert isinstance(conf_dict, dict), type(conf_dict)
    conf_data = conf_dict['process'][0]
    assert conf_data['type'] == 'stft', conf_data['type']
    win_length = conf_data['win_length']
    n_fft = conf_data['n_fft']
    n_shift = conf_data['n_shift']
    window = conf_data['window']

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']
    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        load_input_lengths=train_args.load_input_lengths,
        load_output=True, sort_in_input_length=False,
        preprocess_conf=conf,
        preprocess_args={'train': False},
        target_is_mask=target_is_mask,
        target_is_singlech=args.target_is_singlech,
        test_num_mics=args.test_nmics)

    from datetime import datetime
    from espnet.transform.spectrogram import istft
    from espnet.asr.asr_utils import plot_spectrogram
    # from pb_bss_eval.evaluation import InputMetrics, OutputMetrics
    # from eval_ss import eval_STOI
    # from eval_ss import eval_PESQ
    # from eval_ss import eval_SI_measures
    import soundfile as sf
    from asteroid_metrics import get_metrics
    from asteroid_metrics import average_arrays_in_dic
    import matplotlib.pyplot as plt

    if args.test_oracle:
        sample_dir = os.path.join(args.decode_dir, 'enhanced_v2_oracleIRM_' + datetime.today().strftime("%Y_%m_%d"))
    else:
        sample_dir = os.path.join(args.decode_dir, 'enhanced_v2_' + datetime.today().strftime("%Y_%m_%d"))

    if os.path.exists(sample_dir):
        count = 1
        while os.path.exists("%s_v%d" % (sample_dir, count)):
            count += 1
        sample_dir = "%s_v%d" % (sample_dir, count)
    os.mkdir(sample_dir)

    compute_metrics = ['si_sdr', 'sdr', 'sir', 'sar', 'stoi', 'pesq']
    eval_results = {metric: [] for metric in compute_metrics}
    eval_results0 = {metric: [] for metric in compute_metrics}
    sample_count = 0
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) enhanncing ' + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            x, ys = load_inputs_and_targets(batch)
            # x: Tensor(B, T, C, F)
            # x2: Tensor(T, C, F)
            x2 = x[0]

            if args.target_is_singlech:
                # ys2: Tuple[Tensor(T, F), Tensor(T, F)]
                ys2 = tuple(y[0] for y in ys)
            else:
                # ys2: Tuple[Tensor(T, F), Tensor(T, F)]
                ys2 = tuple(y[0][:, 0, :] for y in ys)

            # ys: (num_spkrs, B, ...)
            ys = np.stack(ys, axis=0)

            # xs_enhanced: List[Tensor(T, F), Tensor(T, F)]
            if args.test_oracle:
                xs_enhanced, masks, ilens = model.enhance(x, targets=ys)
            else:
                xs_enhanced, masks, ilens = model.enhance(x)

            # (2, T)
            wav_enh = np.stack([istft(w[0], n_shift, win_length, window) for w in xs_enhanced], axis=0)
            # (2, T)
            wav_ref = np.stack([istft(y, n_shift, win_length, window) for y in ys2], axis=0)
            # (1, T)
            wav_mix = np.stack([istft(x2[:, 0, :], n_shift, win_length, window)], axis=0)

            # evaluation
            metrics_dict = get_metrics(wav_mix, wav_ref, wav_enh,
                                       sample_rate=train_args.fbank_fs,
                                       metrics_list=compute_metrics,
                                       average=False,
                                       compute_permutation=True)

            metrics_dict = {k: v.squeeze() for k, v in metrics_dict.items()}
            str_metrics = '\n'.join(['  {}: {}'.format(k, v.tolist()) for k, v in metrics_dict.items()])
            logging.info(' evaluation results:\n{}'.format(str_metrics))

            avg_metrics_dict = average_arrays_in_dic(metrics_dict)
            for k, v in avg_metrics_dict.items():
                if k.startswith('input_'):
                    eval_results0[k[6:]].append(v)
                else:
                    eval_results[k].append(v)

            if sample_count < 10 and np.random.rand() < 0.1:
                sf.write(os.path.join(sample_dir, name + '_0.wav'), wav_enh[0], train_args.fbank_fs)
                sf.write(os.path.join(sample_dir, name + '_1.wav'), wav_enh[1], train_args.fbank_fs)

                for spk in range(2):
                    plt.figure(figsize=(16, 32))
                    plt.subplot(4, 1, 1)
                    plt.title('Mask')
                    plot_spectrogram(plt, masks[spk][0, :, 0, :].T, fs=train_args.fbank_fs,
                                     mode='linear', frame_shift=frame_shift,
                                     bottom=False, labelbottom=False)

                    plt.subplot(4, 1, 2)
                    plt.title('Noisy speech')
                    plot_spectrogram(plt, x2[:, 0, :].T, fs=train_args.fbank_fs,
                                     mode='db', frame_shift=frame_shift,
                                     bottom=False, labelbottom=False)

                    plt.subplot(4, 1, 3)
                    plt.title('Clean speech')
                    plot_spectrogram(
#                        plt, ys2[perm[spk]].T,
                        plt, ys2[spk].T,
                        frame_shift=frame_shift,
                        fs=train_args.fbank_fs, mode='db', bottom=False, labelbottom=False)

                    plt.subplot(4, 1, 4)
                    plt.title('Enhanced speech')
                    plot_spectrogram(plt, xs_enhanced[spk][0].T, fs=train_args.fbank_fs,
                                     mode='db', frame_shift=frame_shift)

                    plt.savefig(os.path.join(sample_dir, name + '_{}.png'.format(spk + 1)))
                    plt.clf()

    mean_metrics = {k: np.mean(v) for k, v in eval_results.items()}
    logging.warning('Evaluation of Separated wavs')
    for k, v in eval_results.items():
        logging.warning('mean {}: {}'.format(k.replace('_', '-').upper(), float(np.mean(v))))
    print("\n", flush=True)

    logging.warning('Evaluation of Original Mixture')
    for k, v in eval_results0.items():
        logging.warning('mean {}: {}'.format(k.replace('_', '-').upper(), float(np.mean(v))))
    print("\n", flush=True)
