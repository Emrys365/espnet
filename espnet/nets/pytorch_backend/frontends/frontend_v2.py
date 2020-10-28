from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import logging
import numpy
from pytorch_wpe import wpe_one_iteration
import torch
import torch.nn as nn
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.frontends.beamformer import apply_beamforming_vector
from espnet.nets.pytorch_backend.frontends.beamformer import get_mvdr_vector
from espnet.nets.pytorch_backend.frontends.beamformer import get_mvdr_vector_with_atf
from espnet.nets.pytorch_backend.frontends.beamformer import (
    get_power_spectral_density_matrix,  # noqa: H301
)
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_covariances
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_WPD_filter_conj_with_atf
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import perform_WPD_filtering
from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import signal_framing
from espnet.nets.pytorch_backend.frontends.dnn_beamformer import AttentionReference
from espnet.nets.pytorch_backend.frontends.frontend_wpd_v5 import get_WPD_filter_conj
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class Frontend(nn.Module):
    def __init__(self,
                 idim: int,
                 use_vad_mask: bool = False,
                 # WPE options
                 use_wpe: bool = False,
                 taps: int = 5,
                 delay: int = 3,
                 use_dnn_mask_for_wpe: bool = True,

                 # Beamformer options
                 use_beamformer: bool = False,
                 btype: str = 'blstm',
                 blayers: int = 3,
                 bunits: int = 600,
                 bprojs: int = 600,
                 bnmask: int = 6,
                 badim: int = 320,
                 ref_channel: int = 0,
                 beamformer_type='mvdr',
                 atf_iterations=2,
                 bdropout_rate=0.0,
                 use_beamforming_first=False):
        super().__init__()

        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe

        self.use_beamforming_first = use_beamforming_first
        assert bnmask == 6, bnmask

        self.use_vad_mask = use_vad_mask
        if not self.use_vad_mask:
            from espnet.nets.pytorch_backend.frontends.mask_estimator_v2 import MaskEstimator
            logging.warning('Using normal T-F masks')
        else:
            from espnet.nets.pytorch_backend.frontends.mask_estimator_vad import MaskEstimator
            logging.warning('Using VAD-like masks (same value for all frequencies in each frame)')

        self.mask = MaskEstimator(
            idim, btype, blayers, bunits, bprojs, bdropout_rate, nmask=bnmask
        )
        self.nmask = bnmask
        self.atf_iterations = atf_iterations

        if self.use_wpe:
            self.taps = taps
            self.delay = delay
            self.inverse_power = True
            self.use_dnn_mask_for_wpe = use_dnn_mask_for_wpe
            self.normalization = False
            if self.use_dnn_mask_for_wpe:
                # Use DNN for power estimation
                # (Not observed significant gains)
                self.iterations = 1
                logging.warning('Using {}-iteration DNN-WPE'.format(self.iterations))
            else:
                # Performing as conventional WPE, without DNN Estimator
                self.iterations = 2
                logging.warning('Using {}-iteration Nara-WPE'.format(self.iterations))

        if self.use_beamformer:
            self.ref = AttentionReference(idim, badim) if ref_channel < 0 else None
            self.ref_channel = ref_channel
            logging.warning('Ref channel is {}'.format(ref_channel))
            if beamformer_type not in ('mvdr', 'mpdr', 'wmpdr', 'mvdr_souden', 'mpdr_souden', 'wmpdr_souden', 'wpd_souden', 'wpd'):
                raise ValueError(
                    "Not supporting beamformer_type={}".format(beamformer_type)
                )
            self.beamformer_type = beamformer_type
            if beamformer_type.startswith('wpd'):
                self.btaps = taps
                self.bdelay = delay

    def wpe(self, data: ComplexTensor, ilens: torch.LongTensor, irms=None) \
            -> Tuple[ComplexTensor, torch.LongTensor, ComplexTensor]:
        """The forward function

        Notation:
            B: Batch
            C: Channel
            T: Time or Sequence length
            F: Freq or Some dimension of the feature vector

        Args:
            data: (B, C, T, F)
            ilens: (B,)
        Returns:
            data: (B, C, T, F)
            ilens: (B,)
        """
        # (B, T, C, F) -> (B, F, C, T)
        enhanced = data = data.permute(0, 3, 2, 1).double()

        for i in range(self.iterations):
            # Calculate power: (..., C, T)
            power = enhanced.real ** 2 + enhanced.imag ** 2
            if i == 0 and self.use_dnn_mask_for_wpe:
                # mask: (B, F, C, T)
                assert irms is not None
                if isinstance(irms, list):
                    mask = irms[0].clamp(min=1e-6).double()
                else:
                    mask = irms.clamp(min=1e-6).double()
                if self.normalization:
                    # Normalize along T
                    mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-15)
                # (..., C, T) * (..., C, T) -> (..., C, T)
                power = power * mask

            # Averaging along the channel axis: (..., C, T) -> (..., T)
            power = power.mean(dim=-2)

            # enhanced: (..., C, T) -> (..., C, T)
            enhanced = wpe_one_iteration(
                data.contiguous(), power,
                taps=self.taps, delay=self.delay,
                inverse_power=self.inverse_power)

            enhanced.masked_fill_(make_pad_mask(ilens, enhanced.real), 0)

        # (B, F, C, T) -> (B, T, C, F)
        enhanced = enhanced.permute(0, 3, 2, 1)
        return enhanced, ilens, power

    def beamforming(
        self, data: ComplexTensor, ilens: torch.LongTensor, power=None, irms=None
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

        def apply_beamforming_souden(data, ilens, psd_speech, psd_noise):
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

        def apply_beamforming(data, ilens, psd_n, psd_speech, psd_noise):
            # u: (B, C)
            if self.ref_channel < 0:
                u = self.ref(psd_speech.float(), ilens)[0].double()
            else:
                # (optional) Create onehot vector for fixed reference microphone
#                u = torch.zeros(
#                    *(data.size()[:-3] + (data.size(-2),)), device=data.device
#                )
#                u[..., self.ref_channel].fill_(1)
                u = self.ref_channel

            ws = get_mvdr_vector_with_atf(
                psd_n, psd_speech, psd_noise,
                iterations=self.atf_iterations,
                reference_vector=u,
                normalize_ref_channel=u
            )
            enhanced = apply_beamforming_vector(ws, data)

            return enhanced, ws

        def apply_wpd_beamforming_souden(data, ilens, psd_speech, psd_observed):
            # u: (B, C)
            if self.ref_channel < 0:
                u, _ = self.ref(psd_speech.float(), ilens)
            else:
                # (optional) Create onehot vector for fixed reference microphone
                u = torch.zeros(
                    *(data.size()[:-3] + (data.size(-2),)), device=data.device
                )
                u[..., self.ref_channel].fill_(1)

            ws = get_WPD_filter_conj(psd_observed, psd_speech, u.double())
            enhanced = perform_WPD_filtering(data, ws, self.bdelay, self.btaps)

            return enhanced, ws

        def apply_wpd_beamforming(data, ilens, psd_observed_bar, psd_speech, psd_noise):
            # u: (B, C)
            if self.ref_channel < 0:
                u = self.ref(psd_speech.float(), ilens)[0].double()
            else:
                # (optional) Create onehot vector for fixed reference microphone
#                u = torch.zeros(
#                    *(data.size()[:-3] + (data.size(-2),)), device=data.device
#                )
#                u[..., self.ref_channel].fill_(1)
                u = self.ref_channel

            ws = get_WPD_filter_conj_with_atf(
                psd_observed_bar, psd_speech, psd_noise,
                iterations=self.atf_iterations,
                reference_vector=u,
                normalize_ref_channel=u
            )
            enhanced = perform_WPD_filtering(data, ws, self.bdelay, self.btaps)

            return enhanced, ws

        # data (B, T, C, F) -> (B, F, C, T)
        data = data.permute(0, 3, 2, 1).double()

        # mask: (B, F, C, T)
        assert irms is not None
        masks = [m.double().clamp(min=1e-2) for m in irms]

        if len(masks) == 2:  # (mask_speech, mask_noise)
            mask_speech, mask_noise = masks

            # covariance of source speech
            psd_speech = get_power_spectral_density_matrix(data, mask_speech)
            if self.beamformer_type in ('mvdr', 'mvdr_souden'):
                # covariance of noise
                psd_noise = get_power_spectral_density_matrix(data, mask_noise)
            elif self.beamformer_type == 'mpdr':
                # covariance of observed speech
                psd_observed = FC.einsum('...ct,...et->...ce', [data, data.conj()])
                # covariance of noise
                psd_noise = get_power_spectral_density_matrix(data, mask_noise)
            elif self.beamformer_type == 'mpdr_souden':
                # covariance of observed speech
                psd_observed = FC.einsum('...ct,...et->...ce', [data, data.conj()])
            elif self.beamformer_type == 'wmpdr':
                # covariance of observed speech
                inverse_power = 1 / torch.clamp(power, min=1e-6)
                psd_observed = FC.einsum('...ct,...et->...ce', [data * inverse_power[..., None, :], data.conj()])
                # covariance of noise
                psd_noise = get_power_spectral_density_matrix(data, mask_noise)
            elif self.beamformer_type == 'wmpdr_souden':
                # covariance of observed speech
                inverse_power = 1 / torch.clamp(power, min=1e-6)
                psd_observed = FC.einsum('...ct,...et->...ce', [data * inverse_power[..., None, :], data.conj()])
            elif self.beamformer_type == 'wpd':
                # covariance of noise
                psd_noise = get_power_spectral_density_matrix(data, mask_noise)
                # covariance of stacked observation
                inverse_power = 1 / torch.clamp(power, min=1e-6)
                psd_observed_bar = get_covariances(data, inverse_power, self.bdelay, self.btaps, get_vector=False)
            elif self.beamformer_type == 'wpd_souden':
                # covariance of stacked observation
                inverse_power = 1 / torch.clamp(power, min=1e-6)
                psd_observed_bar = get_covariances(data, inverse_power, self.bdelay, self.btaps, get_vector=False)
            else:
                raise ValueError('Not supporting beamformer_type={}'.format(self.beamformer_type))

            if self.beamformer_type == 'mvdr_souden':
                enhanced, ws = apply_beamforming_souden(data, ilens, psd_speech, psd_noise)
            elif self.beamformer_type in ('mpdr_souden', 'wmpdr_souden'):
                enhanced, ws = apply_beamforming_souden(data, ilens, psd_speech, psd_observed)
            elif self.beamformer_type == 'mvdr':
                enhanced, ws = apply_beamforming(data, ilens, psd_noise, psd_speech, psd_noise)
            elif self.beamformer_type in ('mpdr', 'wmpdr'):
                enhanced, ws = apply_beamforming(data, ilens, psd_observed, psd_speech, psd_noise)
            elif self.beamformer_type == 'wpd_souden':
                enhanced, ws = apply_wpd_beamforming_souden(data, ilens, psd_speech, psd_observed_bar)
            elif self.beamformer_type == 'wpd':
                enhanced, ws = apply_wpd_beamforming(data, ilens, psd_observed_bar, psd_speech, psd_noise)

            # (..., F, T) -> (..., T, F)
            enhanced = enhanced.transpose(-1, -2)
        else:  # multi-speaker case: (mask_speech1, ..., mask_noise)
            assert 4 == len(masks), len(masks)

            # multi-speaker case: (mask_speech1, ..., mask_noise)
            mask_speech = masks[::2]
            mask_noise = masks[1::2]

            # covariance of source speech
            psd_speeches = [
                get_power_spectral_density_matrix(data, mask) for mask in mask_speech
            ]
            if self.beamformer_type in ('mvdr', 'mvdr_souden'):
                # covariance of noise
                psd_noise = [
                    get_power_spectral_density_matrix(data, maskn)
                    for maskn in mask_noise
                ]
            elif self.beamformer_type == 'mpdr':
                # covariance of observed speech
                psd_observed = FC.einsum('...ct,...et->...ce', [data, data.conj()])
                # covariance of noise
                psd_noise = [
                    get_power_spectral_density_matrix(data, maskn)
                    for maskn in mask_noise
                ]
            elif self.beamformer_type == 'mpdr_souden':
                # covariance of observed speech
                psd_observed = FC.einsum('...ct,...et->...ce', [data, data.conj()])
            elif self.beamformer_type == 'wmpdr_souden':
                # covariance of observed speech
                inverse_power = [1 / torch.clamp(p, min=1e-6) for p in power]
                psd_observed = [
                    FC.einsum('...ct,...et->...ce', [data * inv_p[..., None, :], data.conj()])
                    for inv_p in inverse_power
                ]
            elif self.beamformer_type == 'wmpdr':
                # covariance of observed speech
                inverse_power = [1 / torch.clamp(p, min=1e-6) for p in power]
                psd_observed = [
                    FC.einsum('...ct,...et->...ce', [data * inv_p[..., None, :], data.conj()])
                    for inv_p in inverse_power
                ]
                # covariance of noise
                psd_noise = [
                    get_power_spectral_density_matrix(data, maskn)
                    for maskn in mask_noise
                ]
            elif self.beamformer_type == 'wpd':
                # covariance of noise
                psd_noise = [
                    get_power_spectral_density_matrix(data, maskn)
                    for maskn in mask_noise
                ]
                # covariance of stacked observation
                inverse_power = [1 / torch.clamp(p, min=1e-6) for p in power]
                psd_observed_bar = [
                    get_covariances(data, inv_p, self.bdelay, self.btaps, get_vector=False)
                    for inv_p in inverse_power
                ]
            elif self.beamformer_type == 'wpd_souden':
                # covariance of stacked observation
                inverse_power = [1 / torch.clamp(p, min=1e-6) for p in power]
                psd_observed_bar = [
                    get_covariances(data, inv_p, self.bdelay, self.btaps, get_vector=False)
                    for inv_p in inverse_power
                ]
            else:
                raise ValueError('Not supporting beamformer_type={}'.format(self.beamformer_type))

            enhanced = []
            ws = []
            for i, psd_speech in enumerate(psd_speeches):
                # treat all other speakers' psd_speech as noises
                if self.beamformer_type == 'mvdr':
                    enh, w = apply_beamforming(
                        data, ilens, psd_noise[i], psd_speech, psd_noise[i]
                    )
                elif self.beamformer_type == 'mvdr_souden':
                    enh, w = apply_beamforming_souden(
                        data, ilens, psd_speech, psd_noise[i]
                    )
                elif self.beamformer_type == 'mpdr':
                    enh, w = apply_beamforming(
                        data, ilens, psd_observed, psd_speech, psd_noise[i]
                    )
                elif self.beamformer_type == 'mpdr_souden':
                    enh, w = apply_beamforming_souden(
                        data, ilens, psd_speech, psd_observed
                    )
                elif self.beamformer_type == 'wmpdr':
                    enh, w = apply_beamforming(
                        data, ilens, psd_observed[i], psd_speech, psd_noise[i]
                    )
                elif self.beamformer_type == 'wmpdr_souden':
                    enh, w = apply_beamforming_souden(
                        data, ilens, psd_speech, psd_observed[i]
                    )
                elif self.beamformer_type == 'wpd':
                    enh, w = apply_wpd_beamforming(data, ilens, psd_observed_bar[i], psd_speech, psd_noise[i])
                elif self.beamformer_type == 'wpd_souden':
                    enh, w  = apply_wpd_beamforming_souden(data, ilens, psd_speech, psd_observed_bar[i])
                else:
                    raise ValueError('Not supporting beamformer_type={}'.format(self.beamformer_type))

                # (..., F, T) -> (..., T, F)
                enh = enh.transpose(-1, -2)
                enhanced.append(enh)
                ws.append(w)

        return enhanced, ilens

    def forward(self, x: ComplexTensor,
                ilens: Union[torch.LongTensor, numpy.ndarray, List[int]],
                masks=None)\
            -> Tuple[ComplexTensor, torch.LongTensor, Optional[ComplexTensor]]:
        assert len(x) == len(ilens), (len(x), len(ilens))
        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f'Input dim must be 3 or 4: {x.dim()}')
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(numpy.asarray(ilens)).to(x.device)

        mask = [None for n in range(self.nmask)]
        h = x
        if h.dim() == 4:
            if self.training:
                choices = []
                if self.use_wpe and self.use_beamformer:
                    choices.append((True, True))

                #if self.use_wpe:
                #    choices.append((True, False))

                if self.use_beamformer:
                    choices.append((False, True))

                use_wpe, use_beamformer = \
                    choices[numpy.random.randint(len(choices))]

            else:
                use_wpe = self.use_wpe and self.taps > 0
                use_beamformer = self.use_beamformer

            if masks is not None:
                assert len(masks) == 6, len(masks)
                wpe_masks = masks[:2]
                beamforming_masks = masks[2:]
                mask = masks
            else:
                data = h.permute(0, 3, 2, 1)
                mask, _ = self.mask(data.float(), ilens)
                wpe_masks = mask[:2]
                beamforming_masks = mask[2:]

            if self.use_beamforming_first and use_beamformer:
                # 1. Beamformer
                # h: (B, T, C, F) -> h: (B, T, F)
                h, _ = self.beamforming(h, ilens, irms=beamforming_masks)

                # 2. WPE
                if use_wpe:
                    # h: (B, T, C, F) -> h: (B, T, C, F)
                    if isinstance(h, list):
                        # (B, T, F) -> (B, T, C=1, F)
                        h = [hspk[..., None, :] for hspk in h]
                
                        for i, hspk in enumerate(h):
                            h[i], _, _ = self.wpe(hspk, ilens, irms=wpe_masks[i])

                        # (B, T, C=1, F) -> (B, T, F)
                        h = [hspk[..., 0, :] for hspk in h]
                    else:
                        hspks = []
                        for wpe_mask in wpe_masks:
                            hspk, _, _ = self.wpe(h, ilens, irms=wpe_mask)
                            hspks.append(hspk)
                        h = hspks
            else:
                # 1. WPE
                if use_wpe:
                    # h: (B, T, C, F) -> h: (B, T, C, F)
                    hspks = []
                    powers = []
                    for wpe_mask in wpe_masks:
                        hspk, _, power = self.wpe(h, ilens, irms=wpe_mask)
                        hspks.append(hspk)
                        powers.append(power)
                    h = hspks
                else:
                    power_input = (h.real ** 2 + h.imag ** 2).permute(0, 3, 2, 1).double()
                    powers = [
                        # Averaging along the channel axis: (..., C, T) -> (..., T)
                        (power_input * m.clamp(min=1e-6).double()).mean(dim=-2)
                        for m in wpe_masks
                    ]

                # 2. Beamformer
                if use_beamformer:
                    # h: (B, T, C, F) -> h: (B, T, F)
                    if isinstance(h, list):
                        for i, hspk in enumerate(h):
                            h[i], _ = self.beamforming(hspk, ilens, power=powers[i], irms=beamforming_masks[i::2])
                    else:
                        h, _ = self.beamforming(h, ilens, power=powers, irms=beamforming_masks)

            mask = [m.transpose(-1, -3) for m in mask]

        if isinstance(h, list):
            h = [hh.float() for hh in h]
        else:
            h = h.float()
        return h, ilens, mask


def frontend_for(args, idim):
    return Frontend(
        idim=idim,
        use_vad_mask=getattr(args, "use_vad_mask", False),
        # WPE options
        use_wpe=args.use_wpe,
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
        atf_iterations=getattr(args, "atf_iterations", 2),
        bdropout_rate=args.bdropout_rate,
        use_beamforming_first=args.use_beamforming_first)
