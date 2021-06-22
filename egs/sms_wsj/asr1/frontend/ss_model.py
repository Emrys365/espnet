#!/usr/bin/env python3

"""
This script is used to construct End-to-End models of multi-speaker ASR.

Copyright 2017 Johns Hopkins University (Shinji Watanabe)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import division

import argparse
from collections import defaultdict
from distutils.version import LooseVersion
import logging
import math
import os
import sys
import yaml

import chainer
import numpy as np
import six
import torch
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.e2e_asr_mix_transformer import E2E as E2E_ASR_Mix
from espnet.nets.pytorch_backend.e2e_asr_mix_transformer_ss import _create_mask_label
from espnet.nets.pytorch_backend.e2e_asr_mix_transformer_ss import compute_enh_loss
from espnet.nets.pytorch_backend.frontends.stft import Stft
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor

CTC_LOSS_THRESHOLD = 10000


# -*- coding: utf-8 -*-

"""Network related utility tools."""


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss):
        """Define reporter."""
        reporter.report({'loss': loss}, self)


class E2E(E2E_ASR_Mix):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    @staticmethod
    def add_arguments(parser):
        E2E_ASR_Mix.add_arguments(parser)

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Initialize multi-speaker E2E module."""
        # torch.nn.Module.__init__(self)
        super().__init__(idim, odim, args, ignore_id=ignore_id)

        self.normalization = getattr(args, "normalization", False)
        self.reporter = Reporter()

        self.target_is_mask = getattr(args, "target_is_mask", False)  # use IRM as training target
        mask_loss = getattr(args, "mask_loss", "mse")
        if mask_loss == "mse":
            self.loss_func = F.mse_loss
        elif mask_loss == "l1":
            self.loss_func = F.l1_loss
        elif mask_loss == "smooth_l1":
            self.loss_func = F.smooth_l1_loss
        else:
            self.loss_func = None

        self.enh_loss_type = getattr(args, "enh_loss_type", "mask_mse")
        self.mask_type= getattr(args, "mask_type", "cIRM")
        self.ref_channel = args.ref_channel

        with open(args.preprocess_conf) as file:
            preproc_conf = yaml.load(file, Loader = yaml.FullLoader)
            preproc_conf = preproc_conf['process'][0]
        self.stft = Stft(
            win_length=preproc_conf['win_length'],
            n_fft=preproc_conf['n_fft'],
            hop_length=preproc_conf['n_shift'],
            window=preproc_conf['window'],
        )

        self.history = defaultdict(list)

    def forward(self, xs_pad, ilens, ys_pad, wav_lens=None):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, C, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of tuple of target speech of each speaker
                                    (B, num_spkrs, Tmax, F) or (B, num_spkrs, Tmax, C, F)
        :param torch.Tensor wav_lens: batch of lengths of original input waveform sequences (B)
        :return: MSE loss value
        :rtype: torch.Tensor
        """
        if self.frontend is None:
            raise ValueError('Frontend must be used!')

        xs_pad = to_device(self, to_torch_tensor(xs_pad))  # (B, Tmax, C, F)
        # (num_spkrs, B, Tmax, F); or (num_spkrs, B, Tmax, C, F) when self.target_is_mask = True
        ys_pad = to_device(self, to_torch_tensor(ys_pad)).transpose(0, 1)
        # logging.warning('ys_pad: {}, xs_pad: {}'.format(ys_pad.shape, xs_pad.shape))

        # hs_pad: List[ComplexTensor(B, Tmax, F), ..., ComplexTensor(B, Tmax, F)]
        # len(hs_pad) = num_spkrs
        # mask: List[Tensor(B, Tmax, C, F), Tensor(B, Tmax, C, F)]

        # use IRM as training target
        if self.target_is_mask:
            # ys_pad = ys_pad.abs()
            if ys_pad.dim() == 4:
                # --> (num_spkrs, B, Tmax, 1, F)
                ys_pad = ys_pad.unsqueeze(-2)
            with torch.no_grad():
                irm = _create_mask_label(xs_pad, ys_pad, mask_type='PSM^2')
            # targets_sum = xs_pad.abs()
            # with torch.no_grad():
            #     # List[Tensor(B, Tmax, C, F), Tensor(B, Tmax, C, F)]
            #     irm = [ys_pad[spkr] / (targets_sum + 1e-8) for spkr in range(self.num_spkrs)]
            #     irm = [m.clamp(max=1.0) for m in irm]

            if hasattr(self.frontend, 'mask'):
                mask, hlens = self.frontend.mask(xs_pad.permute(0, 3, 2, 1).float(), ilens)
                wpe_masks = [m.permute(0, 3, 2, 1) for m in mask[:2]]
                beamforming_masks = [m.permute(0, 3, 2, 1) for m in mask[2:]]
                mask_speech = beamforming_masks[::2]
                mask_noise = beamforming_masks[1::2]
                wpe_masks[0] = wpe_masks[0].masked_fill(make_pad_mask(hlens, wpe_masks[0], 1), 0.0)
                wpe_masks[1] = wpe_masks[1].masked_fill(make_pad_mask(hlens, wpe_masks[1], 1), 0.0)
                mask_speech[0] = mask_speech[0].masked_fill(make_pad_mask(hlens, mask_speech[0], 1), 0.0)
                mask_speech[1] = mask_speech[1].masked_fill(make_pad_mask(hlens, mask_speech[1], 1), 0.0)
                mask_noise[0] = mask_noise[0].masked_fill(make_pad_mask(hlens, mask_noise[0], 1), 0.0)
                mask_noise[1] = mask_noise[1].masked_fill(make_pad_mask(hlens, mask_noise[1], 1), 0.0)
            else:
                raise NotImplementedError

            # Zero padding
            irm[0] = irm[0].masked_fill(make_pad_mask(hlens, irm[0], 1), 0.0)
            irm[1] = irm[1].masked_fill(make_pad_mask(hlens, irm[1], 1), 0.0)
            #logging.warning('irm: {}, masks: {}'.format([m.shape for m in irm], [m.shape for m in mask]))

            # (B, num_spkrs ** 2)
            loss_perm = torch.stack([self.loss_func(mask_speech[i // self.num_spkrs],
                                                    irm[i % self.num_spkrs],
                                                    reduction='none').mean(dim=(-1, -2, -3))
                                    for i in range(self.num_spkrs ** 2)], dim=1)
            loss_masks, min_perm = self.pit.pit_process(loss_perm)

            batch_size = len(ilens)
            # (num_spkrs, B, Tmax, C, F)
            irm = torch.stack(irm, dim=0)
            for b in range(batch_size):  # B
                irm[:, b] = irm[min_perm[b], b]

            loss_maskn = sum([
                self.loss_func(mask_noise[i], 1 - irm[i], reduction='none').mean()
                for i in range(self.num_spkrs)
            ])
            loss_maskwpe = sum([
                self.loss_func(wpe_masks[i], irm[i], reduction='none').mean()
                for i in range(self.num_spkrs)
            ])
            self.loss = loss_masks + loss_maskn + loss_maskwpe
        else:
            hs_pad, hlens, mask = self.frontend(xs_pad, ilens)
            #print(f'hs_pad: {[h.shape for h in hs_pad]}, ys_pad: {[y.shape for y in ys_pad]}, wav_lens: {wav_lens}', flush=True)
            # Zero padding
            hs_pad[0] = hs_pad[0].masked_fill(
                make_pad_mask(hlens, hs_pad[0].real, 1), 0.0)
            hs_pad[1] = hs_pad[1].masked_fill(
                make_pad_mask(hlens, hs_pad[1].real, 1), 0.0)
            if self.normalization:
                hs_pad[0] = hs_pad[0] / hs_pad[0].detach().abs().max()
                hs_pad[1] = hs_pad[1] / hs_pad[1].detach().abs().max()
                ys_pad[0] = ys_pad[0] / ys_pad[0].abs().max()
                ys_pad[1] = ys_pad[1] / ys_pad[1].abs().max()

            # (B, num_spkrs ** 2)
            loss, min_perm = compute_enh_loss(
                xs_pad, wav_lens, ys_pad, hs_pad, mask, self.enh_loss_type,
                mask_type=self.mask_type, stft=self.stft, ref_channel=self.ref_channel
            )
            #loss_perm = torch.stack([(self.loss_func(hs_pad[i // self.num_spkrs].real,
            #                                     ys_pad[i % self.num_spkrs].real,
            #                                     reduction='none')
            #                        + self.loss_func(hs_pad[i // self.num_spkrs].imag,
            #                                     ys_pad[i % self.num_spkrs].imag,
            #                                     reduction='none')
            #                        + self.loss_func(hs_pad[i // self.num_spkrs].abs(),
            #                                     ys_pad[i % self.num_spkrs].abs(),
            #                                     reduction='none')).mean(dim=(-1, -2))
            #                        for i in range(self.num_spkrs ** 2)], dim=1)
            #loss, min_perm = self.pit.pit_process(loss_perm)
            self.loss = loss

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_data)
        else:
            logging.warning('loss (={}) is not correct'.format(loss_data))

        return self.loss

    def enhance(self, xs, targets=None):
        """Forward only the frontend stage.

        :param ndarray xs: input acoustic feature (B, T, C, F)
        :param ndarray targets: original clean speech spectrum
                                List[Tensor(B, T, C, F), Tensor(B, T, C, F)]
        """
        if self.frontend is None:
            raise RuntimeError('Frontend doesn\'t exist')
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)
        
        # (num_spkrs, B, Tmax, F); or (num_spkrs, B, Tmax, C, F) when self.target_is_mask = True
        # ys_pad = to_torch_tensor(ys_pad).transpose(0, 1)  
        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        if self.target_is_mask and targets is not None:
            with torch.no_grad():
                # Tensor(num_spkrs, B, T, C, F)
                targets_th = to_device(self, to_torch_tensor(targets).float())
                if targets_th.dim() == 3:
                    # (num_spkrs, T, F) --> (num_spkrs, B, T, 1, F)
                    targets_th = targets_th.unsqueeze(-2).unsqueeze(1)
                elif targets_th.dim() == 4:
                    if targets_th[0].shape[0] == xs_pad.shape[0]:
                        # (num_spkrs, B, T, F) --> (num_spkrs, B, T, 1, F)
                        targets_th = targets_th.unsqueeze(-2)
                    else:
                        # (num_spkrs, T, C, F) --> (num_spkrs, B, T, C, F)
                        targets_th = targets_th.unsqueeze(1)

                irms = _create_mask_label(xs_pad, ys_pad, mask_type='PSM^2')
                # targets_sum = xs_pad.abs()
                # irms = [targets_th[i] / (targets_sum + 1e-8) for i in range(self.num_spkrs)]
                # irms = [m.clamp(max=1.0) for m in irms]
            irms[0] = irms[0].masked_fill(make_pad_mask(ilens, irms[0], 1), 0.0)
            irms[1] = irms[1].masked_fill(make_pad_mask(ilens, irms[1], 1), 0.0)
            # (num_spkrs, B, T, C, F) -> (num_spkrs, B, F, C, T)
            irms = [irm.permute(0, 3, 2, 1) for irm in irms]
        else:
            irms = None

        # enhanced: List[Tensor(B, T, F), Tensor(B, T, F)]
        enhanced, hlens, mask = self.frontend(xs_pad, ilens, masks=irms)
        # if self.project:
        #     enhanced[0] = ComplexTensor(self.proj(enhanced[0].real), self.proj(enhanced[0].imag))
        #     enhanced[1] = ComplexTensor(self.proj(enhanced[1].real), self.proj(enhanced[1].imag))
        if prev:
            self.train()

        if isinstance(enhanced, (tuple, list)):
            enhanced = list(enhanced)
            mask = list(mask)
            for idx in range(len(enhanced)):  # number of speakers
                enhanced[idx] = enhanced[idx].cpu().numpy()
                mask[idx] = mask[idx].cpu().numpy()
            return enhanced, mask, ilens
        elif isinstance(mask, (tuple, list)):
            mask = list(mask)
            for idx in range(len(mask)):  # number of speakers
                if mask[idx] is not None:
                    mask[idx] = mask[idx].cpu().numpy()
            return enhanced.cpu().numpy(), mask, ilens
        return enhanced.cpu().numpy(), mask.cpu().numpy(), ilens
