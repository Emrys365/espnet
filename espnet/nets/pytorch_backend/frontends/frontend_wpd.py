#!/usr/bin/env python
# encoding: utf-8

from distutils.version import LooseVersion
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_covariances
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_power_spectral_density_matrix
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_WPD_filter_conj_v2
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import perform_WPD_filtering

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion('1.2')


class Frontend(nn.Module):
    def __init__(self,
                 use_wpe: bool,
                 idim: int,
                 channels: int = 2,
                 use_beamformer: bool = False,
                 btaps: int = 5,
                 bdelay: int = 4,
                 btype: str = 'blstmp',
                 blayers: int = 2,
                 bunits: int = 300,
                 bprojs: int = 320,
                 bnmask: int = 2,
                 badim: int = 320,
                 ref_channel: int = -1,
                 bdropout_rate: float = 0.0,
                 normalization: bool = True):
        super().__init__()

        self.eps = 1e-7
        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe
        self.channels = channels
        self.normalization = normalization

        if self.use_beamformer or self.use_wpe:
            self.mask = MaskEstimator(btype, idim, blayers, bunits, bprojs,
                                      bdropout_rate, nmask=bnmask)

        if self.use_wpe:
            # Use DNN for power estimation
            # (Not observed significant gains)
            self.iterations = 1

        if self.use_beamformer:
            self.ref = AttentionReference(idim, badim)
            self.ref_channel = ref_channel
            self.btaps = btaps
            self.bdelay = bdelay if self.btaps > 0 else 1

    def forward(self, x: ComplexTensor,
                ilens: Union[torch.LongTensor, numpy.ndarray, List[int]],
                masks=None)\
            -> Tuple[ComplexTensor, torch.LongTensor, Optional[ComplexTensor]]:
        assert len(x) == len(ilens), (len(x), len(ilens))

        def get_ref_vec(psd_speech):
            # psd_speech: (B, F, C, C)
            B, Fdim, C = psd_speech.shape[:3]
            if self.ref_channel < 0:
                u = self.ref(psd_speech)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros((B, C),
                                device=psd_speech.device)
                u[..., self.ref_channel].fill_(1)
            # u: (B, C) --> (B, C * (btaps + 1))
            # return F.pad(u, (0, self.btaps * C), 'constant', 0)
            return u

        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f'Input dim must be 3 or 4: {x.dim()}')
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(numpy.asarray(ilens)).to(x.device)

        h = x
        WPD_filter_conj1, WPD_filter_conj2 = None, None
        mask_speech1, mask_speech2 = None, None
        if h.dim() != 4:
            return h, ilens, [mask_speech1, mask_speech2]

        # data (B, T, C, F) -> (B, F, C, T)
        Y = h.permute(0, 3, 2, 1)
        C, T = Y.shape[-2:]

        # 0. Estimating masks for speech1 and speech2
        # Args:
        #   h (ComplexTensor): (B, F, C, T)
        #   ilens (torch.Tensor): (B,)
        # Return:
        #   mask: (B, F, C, T)
        (mask_speech1, mask_speech2), _ = self.mask(Y, ilens)

        # Calculate power: (..., C, T)
        power = Y.real ** 2 + Y.imag ** 2
        if masks is not None:
            # List[Tensor(B, T, C, F), Tensor(B, T, C, F)]
            assert len(masks) == 2
            mask_speech1, mask_speech2 = masks[0], masks[1]
        else:
            if self.normalization:
                # Normalize along T
                mask_speech1 = mask_speech1 / mask_speech1.sum(dim=-1).unsqueeze(-1)
                mask_speech2 = mask_speech2 / mask_speech2.sum(dim=-1).unsqueeze(-1)
        # (..., C, T) * (..., C, T) -> (..., C, T)
        power_speech1 = power * mask_speech1
        power_speech2 = power * mask_speech2

        # Averaging along the channel axis: (B, F, C, T) -> (B, F, T)
        power_speech1 = power_speech1.mean(dim=-2)
        power_speech2 = power_speech2.mean(dim=-2)
        inverse_power1 = 1 / torch.clamp(power_speech1, min=self.eps)
        inverse_power2 = 1 / torch.clamp(power_speech2, min=self.eps)

        # covariance matrix: (B, F, (btaps+1) * C, (btaps+1) * C)
        covariance_matrix1 = get_covariances(Y, inverse_power1, self.bdelay, self.btaps, get_vector=False)
        covariance_matrix2 = get_covariances(Y, inverse_power2, self.bdelay, self.btaps, get_vector=False)

        # PSD of speech from each speaker: (B, F, C, C)
        psd_speech1 = get_power_spectral_density_matrix(Y, mask_speech1, normalization=False)
        psd_speech2 = get_power_spectral_density_matrix(Y, mask_speech2, normalization=False)

        # reference vector: (B, C)
        u1 = get_ref_vec(psd_speech1)
        u2 = get_ref_vec(psd_speech2)

        # (B, F, (btaps+1) * C)
        WPD_filter_conj1 = get_WPD_filter_conj_v2(covariance_matrix1, psd_speech1, u1)
        WPD_filter_conj2 = get_WPD_filter_conj_v2(covariance_matrix2, psd_speech2, u2)

        # (B, F, T)
        enhanced1 = perform_WPD_filtering(Y, WPD_filter_conj1, self.bdelay, self.btaps)
        enhanced2 = perform_WPD_filtering(Y, WPD_filter_conj2, self.bdelay, self.btaps)

        # (B, F, T) --> (B, T, F)
        enhanced1 = enhanced1.transpose(-1, -2)
        enhanced2 = enhanced2.transpose(-1, -2)

        h = [enhanced1, enhanced2]

        # (B, F, C, T) -> (B, T, C, F)
        mask_speech1 = mask_speech1.transpose(-1, -3)
        mask_speech2 = mask_speech2.transpose(-1, -3)

        return h, ilens, [mask_speech1, mask_speech2]


def frontend_for(args, idim):
    return Frontend(
        idim=idim,
        # WPE options
        use_wpe=args.use_wpe,
        btaps=args.wpe_taps,
        bdelay=args.wpe_delay,

        # Beamformer options
        use_beamformer=args.use_beamformer,
        btype=args.btype,
        blayers=args.blayers,
        bunits=args.bunits,
        bprojs=args.bprojs,
        badim=args.badim,
        ref_channel=args.ref_channel,
        bdropout_rate=args.bdropout_rate)


################################################################
#                    Customized Beamformer                     #
################################################################
class AttentionReference(torch.nn.Module):
    def __init__(self, bidim, att_dim):
        super().__init__()
        self.mlp_psd = torch.nn.Linear(bidim, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

    def forward(self, psd_in: ComplexTensor,
                scaling: float = 2.0) -> Tuple[torch.Tensor, torch.LongTensor]:
        """The forward function

        Args:
            psd_in (ComplexTensor): (B, F, C, C)
            scaling (float):
        Returns:
            u (torch.Tensor): (B, C)
        """
        B, _, C = psd_in.size()[:3]
        assert psd_in.size(2) == psd_in.size(3), psd_in.size()
        # psd_in: (B, F, C, C)
        datatype = torch.bool if is_torch_1_2_plus else torch.uint8
        psd = psd_in.masked_fill(torch.eye(C, dtype=datatype,
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
        return u
