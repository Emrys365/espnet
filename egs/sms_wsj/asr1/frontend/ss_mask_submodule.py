#!/usr/bin/env python3

"""
This script is used to construct End-to-End models of multi-speaker ASR.

Copyright 2017 Johns Hopkins University (Shinji Watanabe)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from __future__ import division

import logging
import math

import chainer
import numpy as np
import torch
import torch.nn.functional as F
from torch_complex.tensor import ComplexTensor

from chainer import reporter
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.e2e_asr import E2E as E2E_ASR
from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor

CTC_LOSS_THRESHOLD = 10000


class Frontend(torch.nn.Module):
    def __init__(self,
                 idim: int,
                 btype: str = 'blstmp',
                 blayers: int = 2,
                 bunits: int = 300,
                 bprojs: int = 320,
                 bnmasks: int = 3,
                 bdropout_rate: float = 0.0,
                 normalization: bool = False):
        super().__init__()
        self.eps = 1e-7
        self.normalization = normalization
        self.mask = MaskEstimator(btype, idim, blayers, bunits, bprojs,
                                  bdropout_rate, nmask=bnmasks)

    def forward(self, x: ComplexTensor,
                ilens: Union[torch.LongTensor, np.ndarray, List[int]],
                masks=None)\
            -> Tuple[ComplexTensor, torch.LongTensor, Optional[ComplexTensor]]:
        assert len(x) == len(ilens), (len(x), len(ilens))
        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f'Input dim must be 3 or 4: {x.dim()}')
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(np.asarray(ilens)).to(x.device)

        mask_speech1, mask_speech2 = None, None
        if x.dim() != 4:
            return [mask_speech1, mask_speech2]

        # data (B, T, C, F) -> (B, F, C, T)
        Y = x.permute(0, 3, 2, 1)

        # 0. Estimating masks for speech1 and speech2
        # Args:
        #   Y (ComplexTensor): (B, F, C, T)
        #   ilens (torch.Tensor): (B,)
        # Return:
        #   mask: (B, F, C, T)
        (mask_speech1, mask_speech2, mask_noise), _ = self.mask(Y, ilens)

#        if self.normalization:
#            # Normalize along T
#            mask_speech1 = mask_speech1 / mask_speech1.sum(dim=-1).unsqueeze(-1)
#            mask_speech2 = mask_speech2 / mask_speech2.sum(dim=-1).unsqueeze(-1)
            # mask_noise = mask_noise / mask_noise.sum(dim=-1).unsqueeze(-1)

        # (B, F, C, T) -> (B, T, C, F)
        mask_speech1 = mask_speech1.transpose(-1, -3)
        mask_speech2 = mask_speech2.transpose(-1, -3)

        return [mask_speech1, mask_speech2]


"""Network related utility tools."""


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss):
        """Define reporter."""
        reporter.report({'loss': loss}, self)


class PIT(object):
    """Permutation Invariant Training (PIT) module.

    :parameter int num_spkrs: number of speakers for PIT process (2 or 3)
    """

    def __init__(self, num_spkrs):
        """Initialize PIT module."""
        self.num_spkrs = num_spkrs
        if self.num_spkrs == 2:
            self.perm_choices = [[0, 1], [1, 0]]
        elif self.num_spkrs == 3:
            self.perm_choices = [[0, 1, 2], [0, 2, 1], [1, 2, 0], [1, 0, 2], [2, 0, 1], [2, 1, 0]]
        else:
            raise ValueError

    def min_pit_sample(self, loss):
        """Compute the PIT loss for each sample.

        :param 1-D torch.Tensor loss: list of losses for one sample,
            including [h1r1, h1r2, h2r1, h2r2] or [h1r1, h1r2, h1r3, h2r1, h2r2, h2r3, h3r1, h3r2, h3r3]
        :return minimum loss of best permutation
        :rtype torch.Tensor (1)
        :return the best permutation
        :rtype List: len=2

        """
        if self.num_spkrs == 2:
            score_perms = torch.stack([loss[0] + loss[3],
                                       loss[1] + loss[2]]) / self.num_spkrs
        elif self.num_spkrs == 3:
            score_perms = torch.stack([loss[0] + loss[4] + loss[8],
                                       loss[0] + loss[5] + loss[7],
                                       loss[1] + loss[5] + loss[6],
                                       loss[1] + loss[3] + loss[8],
                                       loss[2] + loss[3] + loss[7],
                                       loss[2] + loss[4] + loss[6]]) / self.num_spkrs

        perm_loss, min_idx = torch.min(score_perms, 0)
        permutation = self.perm_choices[min_idx]

        return perm_loss, permutation

    def pit_process(self, losses):
        """Compute the PIT loss for a batch.

        :param torch.Tensor losses: losses (B, 1|4|9)
        :return minimum losses of a batch with best permutation
        :rtype torch.Tensor (B)
        :return the best permutation
        :rtype torch.LongTensor (B, 1|2|3)

        """
        # logging.warning('losses: {}'.format(losses.shape))
        bs = losses.size(0)
        ret = [self.min_pit_sample(losses[i]) for i in range(bs)]

        loss_perm = torch.stack([r[0] for r in ret], dim=0).to(losses.device)  # (B)
        permutation = torch.tensor([r[1] for r in ret]).long().to(losses.device)

        return torch.mean(loss_perm), permutation


class E2E(E2E_ASR, ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2E.encoder_add_arguments(parser)
        E2E.encoder_mix_add_arguments(parser)
        E2E.attention_add_arguments(parser)
        E2E.decoder_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_mix_add_arguments(parser):
        """Add arguments for multi-speaker encoder."""
        group = parser.add_argument_group("E2E encoder setting for multi-speaker")
        # asr-mix encoder
        group.add_argument('--spa', action='store_true',
                           help='Enable speaker parallel attention for multi-speaker speech recognition task.')
        group.add_argument('--elayers-sd', default=4, type=int,
                           help='Number of speaker differentiate encoder layers'
                                'for multi-speaker speech recognition task.')
        return parser

    def __init__(self, args, idim):
        """Initialize multi-speaker E2E module."""
        torch.nn.Module.__init__(self)
        self.verbose = args.verbose
        self.outdir = args.outdir
        self.reporter = Reporter()
        self.num_spkrs = args.num_spkrs
        self.pit = PIT(self.num_spkrs)
        self.project = args.project
        if args.project:
            self.proj = torch.nn.Linear(idim, idim)

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer_sd + args.elayers)
        # subsample = np.ones(args.elayers_sd + args.elayers + 1, dtype=np.int)
        # if args.etype.endswith("p") and not args.etype.startswith("vgg"):
        #     ss = args.subsample.split("_")
        #     for j in range(min(args.elayers_sd + args.elayers + 1, len(ss))):
        #         subsample[j] = int(ss[j])
        # else:
        #     logging.warning(
        #         'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        # logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        # self.subsample = subsample
        self.subsample = [1]

        self.target_is_mask = getattr(args, "target_is_mask", False)  # use IRM as training target
        assert self.target_is_mask == True
        mask_loss = getattr(args, "mask_loss", "mse")
        if mask_loss == "mse":
            self.loss_func = F.mse_loss
        elif mask_loss == "l1":
            self.loss_func = F.l1_loss
        elif mask_loss == "smooth_l1":
            self.loss_func = F.smooth_l1_loss
        else:
            self.loss_func = None

        if getattr(args, "use_frontend", False):  # use getattr to keep compatibility
            self.frontend = Frontend(
                idim=idim,
                # Beamformer options
                btype=args.btype,
                blayers=args.blayers,
                bunits=args.bunits,
                bprojs=args.bprojs,
                bnmasks=args.bnmask,
                bdropout_rate=args.bdropout_rate
            )
            idim = args.n_mels
        else:
            self.frontend = None

        # weight initialization
        self.init_like_chainer()

        self.eps = 1e-8
        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, C, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of tuple of target speech of each speaker
                                    (B, num_spkrs, Tmax, F) or (B, num_spkrs, Tmax, C, F)
        :return: MSE loss value
        :rtype: torch.Tensor
        """
        if self.frontend is None:
            raise ValueError('Frontend must be used!')

        xs_pad = to_torch_tensor(xs_pad)  # (B, Tmax, C, F)
        # (num_spkrs, B, Tmax, F); or (num_spkrs, B, Tmax, C, F) when self.target_is_mask = True
        ys_pad = to_torch_tensor(ys_pad).transpose(0, 1)
        # logging.warning('ys_pad: {}, xs_pad: {}'.format(ys_pad.shape, xs_pad.shape))

        # hs_pad: List[ComplexTensor(B, Tmax, F), ..., ComplexTensor(B, Tmax, F)]
        # len(hs_pad) = num_spkrs
        # mask: List[Tensor(B, Tmax, C, F), Tensor(B, Tmax, C, F)]
        mask = self.frontend(xs_pad, ilens)
        hlens = ilens

        # use power PSM as training target
        if ys_pad.dim() == 4:
            # --> (num_spkrs, B, Tmax, 1, F)
            ys_pad = ys_pad.unsqueeze(-2)
        with torch.no_grad():
            phase_X = xs_pad / (xs_pad.abs() + self.eps)
            phase_Y = [ys_pad[spkr] / (ys_pad[spkr].abs() + self.eps) for spkr in range(self.num_spkrs)]
            cos_theta = [phase_X.real * phase_Y[spkr].real + phase_X.imag * phase_Y[spkr].imag for spkr in range(self.num_spkrs)]
            targets_sum = xs_pad.abs().pow(2)
            # List[Tensor(B, Tmax, C, F), Tensor(B, Tmax, C, F)]
            irm = [ys_pad[spkr].abs().pow(2) / (targets_sum + self.eps) * cos_theta[spkr] for spkr in range(self.num_spkrs)]
            irm = [m.clamp(min=-1, max=1) for m in irm]

        #logging.warning('irm: {}, masks: {}'.format([m.shape for m in irm], [m.shape for m in mask]))
        # Zero padding
        irm[0] = irm[0].masked_fill(make_pad_mask(hlens, irm[0], 1), 0.0)
        irm[1] = irm[1].masked_fill(make_pad_mask(hlens, irm[1], 1), 0.0)
        mask[0] = mask[0].masked_fill(make_pad_mask(hlens, mask[0], 1), 0.0)
        mask[1] = mask[1].masked_fill(make_pad_mask(hlens, mask[1], 1), 0.0)

        # (B, num_spkrs ** 2)
        loss_perm = torch.stack([self.loss_func(mask[i // self.num_spkrs],
                                                irm[i % self.num_spkrs],
                                                reduction='none').mean(dim=(-1, -2, -3))
                                for i in range(self.num_spkrs ** 2)], dim=1)
        loss_masks, min_perm = self.pit.pit_process(loss_perm)
        self.loss = loss_masks

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

        if targets is not None:
            with torch.no_grad():
                # Tensor(num_spkrs, B, T, C, F)
                targets_th = to_device(self, to_torch_tensor(targets).float()).abs()
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
                        
                phase_X = xs_pad / (xs_pad.abs() + self.eps)
                phase_Y = [targets_th[spkr] / (targets_th[spkr].abs() + self.eps) for spkr in range(self.num_spkrs)]
                cos_theta = [phase_X.real * phase_Y[spkr].real + phase_X.imag * phase_Y[spkr].imag for spkr in range(self.num_spkrs)]
                targets_sum = xs_pad.abs().pow(2)
                irms = [targets_th[i].abs().pow(2) / (targets_sum + self.eps) * cos_theta[i] for i in range(self.num_spkrs)]
                irms = [m.clamp(min=-1, max=1) for m in irms]
            irms[0] = irms[0].masked_fill(make_pad_mask(ilens, irms[0], 1), 0.0)
            irms[1] = irms[1].masked_fill(make_pad_mask(ilens, irms[1], 1), 0.0)
            # (num_spkrs, B, T, C, F) -> (num_spkrs, B, F, C, T)
            irms = [irm.permute(0, 3, 2, 1) for irm in irms]

        if prev:
            self.train()

        if isinstance(irms, (tuple, list)):
            irms = list(irms)
            for idx in range(len(irms)):  # number of speakers
                if irms[idx] is not None:
                    irms[idx] = irms[idx].cpu().numpy()
            return irms, ilens
        return irms.cpu().numpy(), ilens
