from distutils.version import LooseVersion
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import logging
import numpy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.frontends.dnn_beamformer import DNN_Beamformer
from espnet.nets.pytorch_backend.frontends.dnn_wpe import DNN_WPE
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_covariances
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_power_spectral_density_matrix
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_stacked_covariance
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import perform_WPD_filtering


is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion('1.2')


def get_WPD_filter_conj(Rf: ComplexTensor,
                        Phi: ComplexTensor,
                        reference_vector: torch.Tensor,
                        eps: float = 1e-15) -> ComplexTensor:
    """Return the WPD (Weighted Power minimization Distortionless response convolutional beamformer) vector:

        h = (Rf^-1 @ Phi_{xx}) @ u / tr[(Rf^-1) @ Phi_{xx}]

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
                 bnmask: int = 2,
                 badim: int = 320,
                 ref_channel: int = -1,
                 beamformer_type='mvdr',
                 bdropout_rate=0.0,
                 use_beamforming_first=False):
        super().__init__()

        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe
        self.use_dnn_mask_for_wpe = use_dnn_mask_for_wpe
        # use frontend for all the data, e.g. in the case of multi-speaker speech separation
        self.use_frontend_for_all = bnmask > 2
        self.use_beamforming_first = use_beamforming_first

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
            self.eps = 1e-7
            self.normalization = False
            self.btaps = 5
            self.bdelay = 3 if self.btaps > 0 else 1
            self.beamformer = DNN_Beamformer(btype=btype,
                                             bidim=idim,
                                             bunits=bunits,
                                             bprojs=bprojs,
                                             blayers=blayers,
                                             bnmask=bnmask,
                                             dropout_rate=bdropout_rate,
                                             badim=badim,
                                             ref_channel=ref_channel,
                                             beamformer_type=beamformer_type)
        else:
            self.beamformer = None

    def forward(self, x: ComplexTensor,
                ilens: Union[torch.LongTensor, numpy.ndarray, List[int]],
                masks=None)\
            -> Tuple[ComplexTensor, torch.LongTensor, Optional[ComplexTensor]]:
        assert len(x) == len(ilens), (len(x), len(ilens))

        def get_ref_vec(psd_speech, ilens):
            # psd_speech: (B, F, C, C)
            B, Fdim, C = psd_speech.shape[:3]
            if self.beamformer.ref_channel < 0:
                u, _ = self.beamformer.ref(psd_speech, ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros((B, C),
                                device=psd_speech.device)
                u[..., self.beamformer.ref_channel].fill_(1)
            # u: (B, C) --> (B, C * (btaps + 1))
            # return F.pad(u, (0, self.btaps * C), 'constant', 0)
            return u

        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f'Input dim must be 3 or 4: {x.dim()}')
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(numpy.asarray(ilens)).to(x.device)

        mask = [None, None]
        h = x
        if h.dim() == 4:
            # Y (B, T, C, F) -> (B, F, C, T)
            Y = h.permute(0, 3, 2, 1)
            C, T = Y.shape[-2:]

            # mask: (B, F, C, T)
            masks, _ = self.beamformer.mask(Y, ilens)
            mask_speech1, mask_speech2, _ = masks

            # Calculate power: (..., C, T)
            power = Y.real ** 2 + Y.imag ** 2
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

            psd_speech1 = get_power_spectral_density_matrix(Y, mask_speech1, normalization=False)
            psd_speech2 = get_power_spectral_density_matrix(Y, mask_speech2, normalization=False)

            # reference vector: (B, C)
            u1 = get_ref_vec(psd_speech1, ilens)
            u2 = get_ref_vec(psd_speech2, ilens)

            # (B, F, (btaps+1) * C)
            WPD_filter_conj1 = get_WPD_filter_conj(covariance_matrix1, psd_speech1, u1)
            WPD_filter_conj2 = get_WPD_filter_conj(covariance_matrix2, psd_speech2, u2)

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
            mask = [mask_speech1, mask_speech2]

        return h, ilens, mask


def frontend_for(args, idim):
    logging.warning('Using WPD5 frontend')
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
        beamformer_type=args.beamformer_type,
        bdropout_rate=args.bdropout_rate,
        use_beamforming_first=args.use_beamforming_first)
