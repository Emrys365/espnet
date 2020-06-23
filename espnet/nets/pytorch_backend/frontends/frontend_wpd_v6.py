#!/usr/bin/env python
# encoding: utf-8

from distutils.version import LooseVersion
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from pytorch_wpe import wpe_one_iteration

import logging
import numpy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_covariances
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_masked_covariance
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_power_spectral_density_matrix
# from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_stacked_covariance
# from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_WPD_filter_conj
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import perform_WPD_filtering

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion('1.2')


def get_stacked_covariance(Y: ComplexTensor,
                           mask: torch.Tensor,
                           btaps: int,
                           normalization=True,
                           eps: float = 1e-15
                           ) -> ComplexTensor:
    """Return PSD matrix of the stacked speech signal x(t,f) = [Y^T(t,f), 0, ..., 0]^T

    Args:
        Y (ComplexTensor): (B, F, C, T)
        mask (torch.Tensor): (B, F, C, T)
        btaps (int):
        normalization (bool):
        eps (float):
    Returns:
        psd (ComplexTensor): (B, F, (btaps + 1) * C, (btaps + 1) * C)
    """
    C = Y.shape[-2]
    # (B, F, C, C)
    psd = get_power_spectral_density_matrix(Y, mask, normalization, eps)
    # (B, F, (btaps + 1) * C, (btaps + 1) * C)
    return FC.pad(psd, (0, btaps * C, 0, btaps * C), 'constant', 0)


def get_WPD_filter_conj(Rf: ComplexTensor,
                        Phi: ComplexTensor,
                        reference_vector: torch.Tensor,
                        eps: float = 1e-15) -> ComplexTensor:
    """Return the WPD (Weighted Power minimization Distortionless response convolutional beamformer) vector:

        h = (Rf^-1 @ Phi_{xx}) / tr[(Rf^-1) @ Phi_{xx}] @ u

    Reference:
        Maximum likelihood convolutional beamformer for simultaneous denoising
        and dereverberation; Nakatani, T. and Kinoshita, K., 2019;
        https://arxiv.org/abs/1908.02710

    Args:
        Rf (ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        Phi (ComplexTensor): (B, F, C, C)
            is speech PSD.
        reference_vector (torch.Tensor): (B, C)
            is the reference_vector.
        eps (float):

    Returns:
        filter_matrix_conj (ComplexTensor): (B, F, (btaps+1) * C)
    """
    C = reference_vector.shape[-1]
    try:
        inv_Rf = Rf.inverse()
    except:
        try:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(Rf.real),
                                             torch.rand_like(Rf.real)) * 1e-4
            Rf = Rf / 10e+4
            Phi = Phi / 10e+4
            Rf += reg_coeff_tensor
            inv_Rf = Rf.inverse()
        except:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(Rf.real),
                                             torch.rand_like(Rf.real)) * 1e-1
            Rf = Rf / 10e+10
            Phi = Phi / 10e+10
            Rf += reg_coeff_tensor
            inv_Rf = Rf.inverse()
    # (B, F, (btaps+1) * C, (btaps+1) * C) --> (B, F, (btaps+1) * C, C)
    inv_Rf_pruned = inv_Rf[..., :C]
    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = FC.einsum('...ec,...cd->...ed', [inv_Rf_pruned, Phi])
    # ws: (..., (btaps+1) * C, C) / (...,) -> (..., (btaps+1) * C, C)
    ws = numerator / (FC.trace(numerator[..., :C, :])[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum('...fec,...c->...fe', [ws, reference_vector])
    # (B, F, (btaps+1) * C)
    return beamform_vector.conj()

'''
def perform_WPD_filtering(Y: ComplexTensor,
                          filter_matrix_conj: ComplexTensor,
                          bdelay: int, btaps: int) \
        -> ComplexTensor:
    """perform_filter_operation

    Args:
        Y : Complex STFT signal with shape (B, F, C, T)
        filter_matrix_conj: Filter matrix (B, F, C)

    Returns:
        enhanced (ComplexTensor): (B, F, T)
    """
    enhanced = FC.einsum('...c,...ct->...t', [filter_matrix_conj, Y])
    return enhanced
#    # (B, F, C, T) --> (B, F, T, C)
#    Ytilde = Y.transpose(-1, -2)
#
#    # (B, F, T, 1)
#    enhanced = FC.matmul(Ytilde, filter_matrix_conj.unsqueeze(-1)).squeeze(-1)
#    return enhanced
'''


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
                 bnmask: int = 3,
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
#             self.beamformer = DNN_Beamformer(bidim=idim,
#                                              badim=badim,
#                                              btaps=btaps,
#                                              bdelay=bdelay,
#                                              ref_channel=ref_channel)
#        else:
#            self.beamformer = None3

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
            return u
            # u: (B, C) --> (B, C * (btaps + 1))
            # return F.pad(u, (0, self.btaps * C), 'constant', 0)

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
        (mask_speech1, mask_speech2, mask_noise), _ = self.mask(Y, ilens)

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
        if self.normalization:
            # Normalize along T
            mask_noise = mask_noise / mask_noise.sum(dim=-1).unsqueeze(-1)
        # (..., C, T) * (..., C, T) -> (..., C, T)
        power_speech1 = power * mask_speech1
        power_speech2 = power * mask_speech2
        power_noise = power * mask_noise

        # Averaging along the channel axis: (B, F, C, T) -> (B, F, T)
        power_speech1 = power_speech1.mean(dim=-2)
        power_speech2 = power_speech2.mean(dim=-2)
        power_noise = power_noise.mean(dim=-2)
        inverse_power1 = 1 / torch.clamp(power_speech1, min=self.eps)
        inverse_power2 = 1 / torch.clamp(power_speech2, min=self.eps)
        inverse_power_noise = 1 / torch.clamp(power_noise, min=self.eps)

        # covariance matrix: (B, F, (btaps+1) * C, (btaps+1) * C)
        covariance_matrix1 = get_masked_covariance(
            Y, inverse_power1, self.bdelay, self.btaps, mask_speech1, normalization=False)
        covariance_matrix2 = get_masked_covariance(
            Y, inverse_power2, self.bdelay, self.btaps, mask_speech2, normalization=False)
        covariance_matrix_noise = get_masked_covariance(
            Y, inverse_power_noise, self.bdelay, self.btaps, mask_noise, normalization=False)

        # stacked speech signal PSD: (B, F, (btaps+1) * C, (btaps+1) * C)
        # psd_ybar1 = get_stacked_covariance(Y, mask_speech1, self.btaps, normalization=False)
        # psd_ybar2 = get_stacked_covariance(Y, mask_speech2, self.btaps, normalization=False)
#        if self.use_wpe:
#            # dereverb: (..., C, T) -> (..., C, T)
#            dereverb_speech1 = wpe_one_iteration(
#                Y.contiguous(), inverse_power1,
#                taps=self.btaps, delay=self.bdelay,
#                inverse_power=False)
#            dereverb_speech2 = wpe_one_iteration(
#                Y.contiguous(), inverse_power2,
#                taps=self.btaps, delay=self.bdelay,
#                inverse_power=False)
#            dereverb_speech1.masked_fill_(make_pad_mask(ilens, dereverb_speech1.real), 0)
#            dereverb_speech2.masked_fill_(make_pad_mask(ilens, dereverb_speech2.real), 0)
#            # PSD of speech from each speaker: (B, F, C, C)
#            psd_ybar1 = get_power_spectral_density_matrix(dereverb_speech1, mask_speech1)
#            psd_ybar2 = get_power_spectral_density_matrix(dereverb_speech2, mask_speech2)
#        else:
            # (B, F, C, C)
        psd_ybar1 = get_power_spectral_density_matrix(Y, mask_speech1, normalization=False)
        psd_ybar2 = get_power_spectral_density_matrix(Y, mask_speech2, normalization=False)

        # reference vector: (B, (btaps+1) * C)
        # (B, C)
        u1 = get_ref_vec(psd_ybar1)
        u2 = get_ref_vec(psd_ybar2)

        # logging.warning('covariance_matrix1: {}, psd_ybar1: {}, psd_speech1: {}, u1: {}'.format(covariance_matrix1.shape, psd_ybar1.shape, psd_speech1.shape, u1.shape))
        # (B, F, (btaps + 1) * C)
        WPD_filter_conj1 = get_WPD_filter_conj(
            covariance_matrix2 + covariance_matrix_noise, psd_ybar1, u1)
        WPD_filter_conj2 = get_WPD_filter_conj(
            covariance_matrix1 + covariance_matrix_noise, psd_ybar2, u2)

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
