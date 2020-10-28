from distutils.version import LooseVersion
from typing import Tuple

import torch
from torch.nn import functional as F
from torch_complex import functional as FC

from espnet.nets.pytorch_backend.frontends.beamformer_old import apply_beamforming_vector
from espnet.nets.pytorch_backend.frontends.beamformer_old import get_mvdr_vector
from espnet.nets.pytorch_backend.frontends.beamformer_old import (
    get_power_spectral_density_matrix,  # noqa: H301
)
#from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator
from torch_complex.tensor import ComplexTensor

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2.0")
is_torch_1_3_plus = LooseVersion(torch.__version__) >= LooseVersion("1.3.0")


class DNN_Beamformer(torch.nn.Module):
    """DNN mask based Beamformer

    Citation:
        Multichannel End-to-end Speech Recognition; T. Ochiai et al., 2017;
        https://arxiv.org/abs/1703.04783

    """

    def __init__(
        self,
        bidim,
        btype="blstmp",
        blayers=3,
        bunits=300,
        bprojs=320,
        bnmask=2,
        dropout_rate=0.0,
        badim=320,
        ref_channel: int = -1,
        beamformer_type="mvdr",
        use_vad_mask: bool = False,
    ):
        super().__init__()
        self.use_vad_mask = use_vad_mask
        if not self.use_vad_mask:
            from espnet.nets.pytorch_backend.frontends.mask_estimator import MaskEstimator
            print('Using normal T-F masks for beamforming', flush=True)
        else:
            from espnet.nets.pytorch_backend.frontends.mask_estimator_vad_v1 import MaskEstimator
            print('Using VAD-like masks for beamforming (same value for all frequencies in each frame)', flush=True)

        self.mask = MaskEstimator(
            btype, bidim, blayers, bunits, bprojs, dropout_rate, nmask=bnmask
        )
        self.ref = AttentionReference(bidim, badim)
        self.ref_channel = ref_channel

        self.nmask = bnmask

        if beamformer_type not in ('mvdr', 'mpdr', 'wmpdr'):
            raise ValueError(
                "Not supporting beamformer_type={}".format(beamformer_type)
            )
        self.beamformer_type = beamformer_type

    def forward(
        self, data: ComplexTensor, ilens: torch.LongTensor, targets=None
    ) -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq

        Args:
            data (ComplexTensor): (B, T, C, F)
            ilens (torch.Tensor): (B,)
        Returns:
            enhanced (ComplexTensor): (B, T, F)
            ilens (torch.Tensor): (B,)

        """

        def apply_beamforming(data, ilens, psd_speech, psd_noise):
            # u: (B, C)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_speech.float(), ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(
                    *(data.size()[:-3] + (data.size(-2),)), device=data.device
                )
                u[..., self.ref_channel].fill_(1)

            ws = get_mvdr_vector(psd_speech, psd_noise, u.double())
            enhanced = apply_beamforming_vector(ws, data)

            return enhanced, ws

        # data (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1).double()

        # mask: (B, F, C, T)
        if targets is not None:
            masks = [m.double() for m in targets]
            assert self.nmask == len(masks)
        else:
            masks, _ = self.mask(data.float(), ilens)
            assert self.nmask == len(masks)
#            masks = [torch.clamp(m, min=1e-2).double() for m in masks]

        if self.nmask == 2:  # (mask_speech, mask_noise)
            mask_speech, mask_noise = masks

            # covariance of source speech
            psd_speech = get_power_spectral_density_matrix(data, mask_speech)
            if self.beamformer_type == 'mvdr':
                # covariance of noise
                psd_noise = get_power_spectral_density_matrix(data, mask_noise)
            elif self.beamformer_type == 'mpdr':
                # covariance of observed speech
                psd_noise = FC.einsum('...ct,...et->...ce', [data, data.conj()])
            elif self.beamformer_type == 'wmpdr':
                # covariance of observed speech
                psd_noise = FC.einsum('...ct,...et->...ce', [data, data.conj()])
            else:
                raise ValueError('Not supporting beamformer_type={}'.format(self.beamformer_type))

            enhanced, ws = apply_beamforming(data, ilens, psd_speech, psd_noise)

            # (..., F, T) -> (..., T, F)
            enhanced = enhanced.transpose(-1, -2)
            mask_speech = mask_speech.transpose(-1, -3)
        else:  # multi-speaker case: (mask_speech1, ..., mask_noise)
            mask_speech = list(masks[:-1])
            mask_noise = masks[-1]

            # covariance of source speech
            psd_speeches = [
                get_power_spectral_density_matrix(data, mask) for mask in mask_speech
            ]
            if self.beamformer_type == 'mvdr':
                # covariance of noise
                psd_noise = get_power_spectral_density_matrix(data, mask_noise)
            elif self.beamformer_type == 'mpdr':
                # covariance of observed speech
                psd_noise = FC.einsum('...ct,...et->...ce', [data, data.conj()])
            else:
                raise ValueError('Not supporting beamformer_type={}'.format(self.beamformer_type))

            enhanced = []
            ws = []
            for i in range(self.nmask - 1):
                psd_speech = psd_speeches.pop(i)
                # treat all other speakers' psd_speech as noises
                if self.beamformer_type == 'mvdr':
                    enh, w = apply_beamforming(
                        data, ilens, psd_speech, sum(psd_speeches) + psd_noise
                    )
                elif self.beamformer_type == 'mpdr':
                    enh, w = apply_beamforming(
                        data, ilens, psd_speech, psd_noise
                    )
                elif self.beamformer_type == 'wmpdr':
                    enh, w = apply_beamforming(
                        data, ilens, psd_speech, psd_noise
                    )
                else:
                    raise ValueError('Not supporting beamformer_type={}'.format(self.beamformer_type))
                psd_speeches.insert(i, psd_speech)

                # (..., F, T) -> (..., T, F)
                enh = enh.transpose(-1, -2)
                mask_speech[i] = mask_speech[i].transpose(-1, -3)

                enhanced.append(enh)
                ws.append(w)

        return enhanced, ilens, mask_speech


class AttentionReference(torch.nn.Module):
    def __init__(self, bidim, att_dim):
        super().__init__()
        self.mlp_psd = torch.nn.Linear(bidim, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

    def forward(
        self, psd_in: ComplexTensor, ilens: torch.LongTensor, scaling: float = 2.0
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
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
        datatype = torch.bool if is_torch_1_3_plus else torch.uint8
        datatype2 = torch.bool if is_torch_1_2_plus else torch.uint8
        psd = psd_in.masked_fill(
            torch.eye(C, dtype=datatype, device=psd_in.device).type(datatype2), 0
        )
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
