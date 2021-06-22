#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import logging
import math
import sys

from chainer import reporter as reporter_module
from chainer import training
from chainer.training.updater import StandardUpdater
import numpy as np
import torch
from torch.nn.parallel import data_parallel

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.utils.training.evaluator import BaseEvaluator


if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest


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
        # xs: Tensor(B, T, C, F)
        # ys: Tuple(Tensor(B, T, F), Tensor(B, T, F))
        if len(batch[0]) == 2:
            xs, ys = batch[0]
            wav_lens = None
        else:
            xs, ys, wav_lens = batch[0]
            wav_lens = torch.as_tensor(wav_lens, device=device)
        # Convert zip object to list in python 3.x
        # ys = list(ys)

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
            # batch of padded input sequences (B, Tmax, idim)
            xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device, dtype=self.dtype)

        ilens = torch.from_numpy(ilens).to(device)
        if ys[0][0].dtype.kind == 'c':
            # (B, Tmax, F)
            ys_pad_spk1_real = pad_list(
                [torch.from_numpy(y.real).float() for y in ys[0]], 0).to(device, dtype=self.dtype)
            ys_pad_spk2_real = pad_list(
                [torch.from_numpy(y.real).float() for y in ys[1]], 0).to(device, dtype=self.dtype)
            ys_pad_spk1_imag = pad_list(
                [torch.from_numpy(y.imag).float() for y in ys[0]], 0).to(device, dtype=self.dtype)
            ys_pad_spk2_imag = pad_list(
                [torch.from_numpy(y.imag).float() for y in ys[1]], 0).to(device, dtype=self.dtype)
            # (B, num_spkrs, Tmax, F)
            ys_pad_real = torch.stack((ys_pad_spk1_real, ys_pad_spk2_real), dim=1)
            ys_pad_imag = torch.stack((ys_pad_spk1_imag, ys_pad_spk2_imag), dim=1)
            # ys_pad_spk1_real = [torch.from_numpy(y[0].real).float().to(device, dtype=self.dtype) for y in ys]
            # ys_pad_spk1_imag = [torch.from_numpy(y[0].imag).float().to(device, dtype=self.dtype) for y in ys]
            # ys_pad_spk2_real = [torch.from_numpy(y[1].real).float().to(device, dtype=self.dtype) for y in ys]
            # ys_pad_spk2_imag = [torch.from_numpy(y[1].imag).float().to(device, dtype=self.dtype) for y in ys]

            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it to E2E here
            # because torch.nn.DataParallel can't handle it.
            ys_pad = {'real': ys_pad_real, 'imag': ys_pad_imag}
            # ys_pad_spk1 = {'real': ys_pad_spk1_real, 'imag': ys_pad_spk1_imag}
            # ys_pad_spk2 = {'real': ys_pad_spk2_real, 'imag': ys_pad_spk2_imag}
        else:
            # (B, Tmax, F)
            ys_pad_spk1 = pad_list(
                [torch.from_numpy(y).float() for y in ys[0]], 0).to(device, dtype=self.dtype)
            ys_pad_spk2 = pad_list(
                [torch.from_numpy(y).float() for y in ys[1]], 0).to(device, dtype=self.dtype)
            # (B, num_spkrs, Tmax, F)
            ys_pad = torch.stack((ys_pad_spk1, ys_pad_spk2), dim=1)
            # ys_pad_spk1 = [torch.from_numpy(y[0]).float().to(device, dtype=self.dtype) for y in ys]
            # ys_pad_spk2 = [torch.from_numpy(y[1]).float().to(device, dtype=self.dtype) for y in ys]

        # logging.warning('ys: {}, ys_pad: {}'.format([[x.shape for x in y] for y in ys], [n.shape for n in ys_pad.values()]))
        return xs_pad, ilens, ys_pad, wav_lens


class CustomEvaluator(BaseEvaluator):
    """Custom Evaluator for Pytorch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        iterator (chainer.dataset.Iterator) : The train iterator.

        target (link | dict[str, link]) :Link object or a dictionary of
            links to evaluate. If this is just a link object, the link is
            registered by the name ``'main'``.
        converter (espnet.asr.pytorch_backend.asr.CustomConverter): Converter
            function to build input arrays. Each batch extracted by the main
            iterator and the ``device`` option are passed to this function.
            :func:`chainer.dataset.concat_examples` is used by default.

        device (torch.device): The device used.
        ngpu (int): The number of GPUs.
    """

    def __init__(self, model, iterator, target, converter, device, updater, ngpu=None):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.converter = converter
        self.device = device
        self.updater = updater
        if ngpu is not None:
            self.ngpu = ngpu
        elif device.type == "cpu":
            self.ngpu = 0
        else:
            self.ngpu = 1

    # The core part of the update routine can be customized by overriding
    def evaluate(self):
        """Main evaluate routine for CustomEvaluator."""
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    x = self.converter(batch, self.device)
                    if self.ngpu == 0:
                        self.model(*x)
                    else:
                        # apex does not support torch.nn.DataParallel
                        data_parallel(self.model, x, range(self.ngpu))

                summary.add(observation)
        self.model.train()

        summary_mean = summary.compute_mean()
        #print('loss:', self.model.history['validation/main/loss'], "summary_mean['validation/main/loss']:", summary_mean['validation/main/loss'], flush=True)
        if len(self.model.history['validation/main/loss']) > 0 and summary_mean['validation/main/loss'] > min(self.model.history['validation/main/loss']):
            # lower the learning rate by half (ReduceLROnPlateau)
            for param_group in self.updater.get_optimizer('main').param_groups:
                print('\nLR: {} -> {}'.format(param_group['lr'], param_group['lr'] * 0.5), flush=True)
                param_group['lr'] = param_group['lr'] * 0.5

        for k, v in summary_mean.items():
            self.model.history[k].append(v)

        return summary_mean


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
                 optimizer, scheduler, converter, device, ngpu,
                 grad_noise=False, accum_grad=1, use_apex=False):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip_threshold = grad_clip_threshold
        if scheduler is not None:
            self.scheduler = scheduler
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.accum_grad = accum_grad
        self.forward_count = 0
        self.grad_noise = grad_noise
        self.iteration = 0
        self.use_apex = use_apex

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

        # update parameters
        self.forward_count += 1
        if self.forward_count != self.accum_grad:
            return
        self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()
        #self.scheduler.step()
        optimizer.zero_grad()

    def update(self):
        self.update_core()
        # #iterations with accum_grad > 1
        # Ref.: https://github.com/espnet/espnet/issues/777
        if self.forward_count == 0:
            self.iteration += 1

