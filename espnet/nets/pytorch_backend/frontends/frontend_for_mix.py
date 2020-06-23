#!/usr/bin/env python
# encoding: utf-8

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator

from espnet.nets.pytorch_backend.frontends.beamformer \
    import apply_beamforming_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_mvdr_vector
from espnet.nets.pytorch_backend.frontends.beamformer \
    import get_power_spectral_density_matrix

from pytorch_wpe import wpe_one_iteration
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class Frontend(nn.Module):
    def __init__(self,
                 idim: int,
                 # WPE options
                 use_wpe: bool = False,
                 wtype: str = 'blstmp',
                 wlayers: int = 3,
                 wunits: int = 300,
                 wprojs: int = 320,
                 wdropout_rate: float = 0.0,
                 taps: int = 5,
                 delay: int = 3,
                 use_dnn_mask_for_wpe: bool = True,

                 # Beamformer options
                 use_beamformer: bool = False,
                 btype: str = 'blstmp',
                 blayers: int = 3,
                 bunits: int = 300,
                 bprojs: int = 320,
                 bnmask: int = 3,
                 badim: int = 320,
                 ref_channel: int = -1,
                 bdropout_rate=0.0,
                 use_beamforming_first=False):
        super().__init__()

        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe
        self.use_dnn_mask_for_wpe = use_dnn_mask_for_wpe
        # use frontend for all the data, e.g. in the case of multi-speaker speech separation
        self.use_frontend_for_all = bnmask > 2
        self.use_beamforming_first = use_beamforming_first

        if self.use_beamformer or self.use_wpe:
            self.mask = MaskEstimator(btype, idim, blayers, bunits, bprojs,
                                  bdropout_rate, nmask=bnmask)

        if self.use_wpe:
            if self.use_dnn_mask_for_wpe:
                # Use DNN for power estimation
                # (Not observed significant gains)
                iterations = 1
            else:
                # Performing as conventional WPE, without DNN Estimator
                iterations = 2

            self.wpe = DNN_WPE(wtype=wtype,
                               widim=idim,
                               wunits=wunits,
                               wprojs=wprojs,
                               wlayers=wlayers,
                               taps=taps,
                               delay=delay,
                               dropout_rate=wdropout_rate,
                               iterations=iterations,
                               use_dnn_mask=use_dnn_mask_for_wpe)
        else:
            self.wpe = None

        if self.use_beamformer:
            self.beamformer = DNN_Beamformer(bidim=idim,
                                             badim=badim,
                                             ref_channel=ref_channel)
        else:
            self.beamformer = None

    def forward(self, x: ComplexTensor,
                ilens: Union[torch.LongTensor, numpy.ndarray, List[int]]) \
            -> Tuple[ComplexTensor, torch.LongTensor, Optional[ComplexTensor]]:
        assert len(x) == len(ilens), (len(x), len(ilens))
        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f'Input dim must be 3 or 4: {x.dim()}')
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(numpy.asarray(ilens)).to(x.device)

        h = x
        ws1, ws2 = None, None
        mask_speech1, mask_speech2 = None, None
        if h.dim() == 4:
            if self.training:
                choices = [(False, False)] if not self.use_frontend_for_all else []
                if self.use_wpe and self.use_beamformer:
                    choices.append((True, True))
                
                if self.use_wpe:
                    choices.append((True, False))

                if self.use_beamformer:
                    choices.append((False, True))

                use_wpe, use_beamformer = \
                    choices[numpy.random.randint(len(choices))]

            else:
                use_wpe = self.use_wpe
                use_beamformer = self.use_beamformer

            # data (B, T, C, F) -> (B, F, C, T)
            h2 = h.permute(0, 3, 2, 1)

            # 0. Pre-Separation using mask
            # Args:
            #   h (ComplexTensor): (B, F, C, T)
            #   ilens (torch.Tensor): (B,)
            # Return:
            #   mask: (B, F, C, T)
            (mask_speech1, mask_speech2, mask_noise), _ = self.mask(h2, ilens)

            # separated speech: (B, F, C, T)
            h_s1 = h2 * mask_speech1
            h_s2 = h2 * mask_speech2

            # 1. WPE
            if use_wpe:
                # h: (B, F, C, T) -> h: (B, F, C, T)
                h_s1, ilens, mask = self.wpe(h_s1, h2, ilens)
                h_s2, ilens, mask = self.wpe(h_s2, h2, ilens)

            # PSD: (B, F, C, C)
            # (..., T, C, C) -> (..., C, C)
            psd_speech1 = get_power_spectral_density_matrix(h_s1, mask_speech1)
            psd_speech2 = get_power_spectral_density_matrix(h_s2, mask_speech2)
            psd_noise = get_power_spectral_density_matrix(h2, mask_noise)

            # 2. Beamformer
            if use_beamformer:
                # h: (B, F, C, T) -> h: (B, F, T)
                h_s1, ilens, ws1 = self.beamformer(h_s1, psd_speech1, psd_noise, ilens)
                h_s2, ilens, ws2 = self.beamformer(h_s2, psd_speech2, psd_noise, ilens)

            if h_s1.dim() == h_s2.dim() == 3:
                # (..., F, T) -> (..., T, F)
                h_s1 = h_s1.transpose(-1, -2)
                h_s2 = h_s2.transpose(-1, -2)
            elif h_s1.dim() == h_s2.dim() == 4:
                # (..., F, C, T) -> (..., T, C, F)
                h_s1 = h_s1.transpose(-1, -3)
                h_s2 = h_s2.transpose(-1, -3)

            # for saving CUDA memory
            ## (B, F, C, T) -> (B, T, C, F)
            #mask_speech1 = mask_speech1.transpose(-1, -3)
            #mask_speech2 = mask_speech2.transpose(-1, -3)
            h = [h_s1, h_s2]

        #return h, ilens, (mask_speech1, mask_speech2)
        # for saving CUDA memory
        return h, ilens, [None, None]


def frontend_for(args, idim):
    return Frontend(
        idim=idim,
        # WPE options
        use_wpe=args.use_wpe,
        wtype=args.wtype,
        wlayers=args.wlayers,
        wunits=args.wunits,
        wprojs=args.wprojs,
        wdropout_rate=args.wdropout_rate,
        taps=args.wpe_taps,
        delay=args.wpe_delay,
        use_dnn_mask_for_wpe=args.use_dnn_mask_for_wpe,

        # Beamformer options
        use_beamformer=args.use_beamformer,
        btype=args.btype,
        blayers=args.blayers,
        bunits=args.bunits,
        bprojs=args.bprojs,
        bnmask=args.bnmask,
        badim=args.badim,
        ref_channel=args.ref_channel,
        bdropout_rate=args.bdropout_rate,
        use_beamforming_first=args.use_beamforming_first)


################################################################
#                    Customized Beamformer                     #
################################################################
class DNN_Beamformer(torch.nn.Module):
    """DNN mask based Beamformer
    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        https://arxiv.org/abs/1703.04783
    """

    def __init__(self,
                 bidim,
                 badim=320,
                 ref_channel: int = None,
                 beamformer_type='mvdr'):
        super().__init__()
        self.ref = AttentionReference(bidim, badim)
        self.ref_channel = ref_channel

        if beamformer_type != 'mvdr':
            raise ValueError(
                'Not supporting beamformer_type={}'.format(beamformer_type))
        self.beamformer_type = beamformer_type

    def forward(self, data: ComplexTensor,
                psd_speech: ComplexTensor,
                psd_noise: ComplexTensor,
                ilens: torch.LongTensor) \
            -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
        """The forward function
        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq
        Args:
            data (ComplexTensor): (B, F, C, T)
            psd_speech (ComplexTensor): (B, F, C, C)
            psd_noise (ComplexTensor): (B, F, C, C)
            ilens (torch.Tensor): (B,)
        Returns:
            enhanced (ComplexTensor): (B, F, T)
            ilens (torch.Tensor): (B,)
        """
        # u: (B, C)
        if self.ref_channel is None:
            u, _ = self.ref(psd_speech, ilens)
        else:
            # (optional) Create onehot vector for fixed reference microphone
            u = torch.zeros(*(data.size()[:-3] + (data.size(-2),)),
                            device=data.device)
            u[..., self.ref_channel].fill_(1)

        ws = get_mvdr_vector(psd_speech, psd_noise, u)
        enhanced = apply_beamforming_vector(ws, data)

        # (..., F, T) -> (..., T, F)
        #enhanced = enhanced.transpose(-1, -2)

        #return enhanced, ilens, ws
        # for saving CUDA memory
        return enhanced, ilens, None


class AttentionReference(torch.nn.Module):
    def __init__(self, bidim, att_dim):
        super().__init__()
        self.mlp_psd = torch.nn.Linear(bidim, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

    def forward(self, psd_in: ComplexTensor, ilens: torch.LongTensor,
                scaling: float = 2.0) -> Tuple[torch.Tensor, torch.LongTensor]:
        """The forward function
        Args:
            psd_in (ComplexTensor): (B, F, C, C)
            ilens (torch.Tensor): (B,)
            scaling (float):
        Returns:
            u (torch.Tensor): (B, C)
            ilens (torch.Tensor): (B,)
        """
        B, _, C = psd_in.size()[:3]
        assert psd_in.size(2) == psd_in.size(3), psd_in.size()
        # psd_in: (B, F, C, C)
        psd = psd_in.masked_fill(torch.eye(C, dtype=torch.bool,
                                           device=psd_in.device), 0)
        # psd: (B, F, C, C) -> (B, C, F)
        psd = (psd.sum(dim=-1) / (C - 1)).transpose(-1, -2)

        # Calculate amplitude
        psd_feat = (psd.real ** 2 + psd.imag ** 2) ** 0.5

        # (B, C, F) -> (B, C, F2)
        mlp_psd = self.mlp_psd(psd_feat)
        # (B, C, F2) -> (B, C, 1) -> (B, C)
        e = self.gvec(torch.tanh(mlp_psd)).squeeze(-1)
        u = F.softmax(scaling * e, dim=-1)
        return u, ilens


################################################################
#                        Customized WPE                        #
################################################################
class DNN_WPE(torch.nn.Module):
    def __init__(self,
                 wtype: str = 'blstmp',
                 widim: int = 257,
                 wlayers: int = 3,
                 wunits: int = 300,
                 wprojs: int = 320,
                 dropout_rate: float = 0.0,
                 taps: int = 5,
                 delay: int = 3,
                 use_dnn_mask: bool = True,
                 iterations: int = 1,
                 normalization: bool = False,
                 ):
        super().__init__()
        self.iterations = iterations
        self.taps = taps
        self.delay = delay

        self.normalization = normalization
        self.use_dnn_mask = use_dnn_mask

        self.inverse_power = True

        if self.use_dnn_mask:
            self.mask_est = MaskEstimator(
                wtype, widim, wlayers, wunits, wprojs, dropout_rate, nmask=1)

    def forward(self, data_sep: ComplexTensor,
                data_mix: ComplexTensor, ilens: torch.LongTensor) \
            -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
        """The forward function
        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq or Some dimension of the feature vector
        Args:
            data_sep: (B, F, C, T)
            data_mix: (B, F, C, T)
            ilens: (B,)
        Returns:
            data: (B, F, C, T)
            ilens: (B,)
        """
        # separated data: (B, F, C, T)
        # The separated speech should be only used for computing the mask.
        enhanced = data_sep
        mask = None

        for i in range(self.iterations):
            # Calculate power: (..., C, T)
            power = enhanced.real ** 2 + enhanced.imag ** 2
            if i == 0 and self.use_dnn_mask:
                # mask: (B, F, C, T)
                (mask,), _ = self.mask_est(enhanced, ilens)
                if self.normalization:
                    # Normalize along T
                    mask = mask / mask.sum(dim=-1)[..., None]
                # (..., C, T) * (..., C, T) -> (..., C, T)
                power = power * mask

            # Averaging along the channel axis: (..., C, T) -> (..., T)
            power = power.mean(dim=-2)

            # enhanced: (..., C, T) -> (..., C, T)
            enhanced = wpe_one_iteration(
                data_mix.contiguous(), power,
                taps=self.taps, delay=self.delay,
                eps=1e-7,
                inverse_power=self.inverse_power)

            enhanced.masked_fill_(make_pad_mask(ilens, enhanced.real), 0)

        #return enhanced, ilens, mask
        # for saving CUDA memory
        return enhanced, ilens, None
