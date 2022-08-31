from itertools import permutations
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


def find_peaks(
    x,
    cyclic: bool = False,
    mph: Optional[float] = None,
    mpd: int = 1,
    threshold: float = 0.0,
    edge: Optional[str] = "rising",
    kpsh: bool = False,
    valley: bool = False,
    sort: str = "value",
) -> torch.Tensor:
    """Find peaks in data based on their amplitude and other features.

    Args:
        x : 1D array_like input data.
        cyclic (bool): True to allow peaks/valleys at the beginning/end of `x`
            by cyclicly repeat sequence `x`
        mph (float):
            if not None, only detect peaks (valleys) that are greater (smaller)
            than minimum peak height (maximum peak height).
        mpd : minimum distance (> 0) between adjacent peaks
            only detect peaks that are at least separated by minimum peak distance
            (in number of data)
        threshold : minimum height difference between a peak and its adjacent samples
            only detect peaks (valleys) that are greater (smaller) than `threshold`
            in relation to their immediate neighbors.
        edge (str): one of (None, 'rising', 'falling', 'both')
            for a flat peak:
                - "rising":  keep only the rising edge
                - "falling": keep only the falling edge
                - "both":    keep both edges
                -  None:     don't detect a flat peak
        kpsh (bool): True to keep peaks with same height even if
            they are closer than `mpd`.
        valley (bool): True to find valleys (local minima) instead of peaks.
        sort (str): how to sort the returned indexes
            "value": sort by the peak values in the descending order
            "index": sort by the index value in the ascending order
    Returns:
        ind (torch.Tensor): 1D array_like indexes of the peaks in `x`.

    References:
        [1] https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/doa/detect_peaks.py
    """

    def in1D(x, labels, invert=False):
        """
        Sub-optimal equivalent to numpy.in1D().
        Hopefully this feature will be properly covered soon
        c.f. https://github.com/pytorch/pytorch/issues/3025
        Snippet by Aron Barreira Bordin
        Args:
            x (Tensor): Tensor to search values in
            labels (Tensor/List): 1D array of values to search for
            invert (bool):
                If True, the values in the returned array are inverted (that is,
                False where an element of `x` is in `labels` and True otherwise).

        Returns:
            Tensor: Boolean tensor y of same shape as x, with y[ind] = True if x[ind] in labels

        Example:
            >>> in1D(torch.FloatTensor([1, 2, 0, 3]), [2, 3])
            FloatTensor([False, True, False, True])
        """
        mapping = torch.zeros_like(x).byte()
        for label in labels:
            mapping = mapping | x.eq(label)
        return mapping if invert else ~mapping

    assert edge in ("rising", "falling", "both", None), edge
    # x = torch.atleast_1d(x).to(dtype=torch.float64)
    x = torch.as_tensor(x, dtype=torch.float64)
    if len(x) < 3:
        # too few samples to find a peak
        return torch.as_tensor([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks (x[1:] - x[:-1])
    # dx = torch.diff(x, n=1)
    dx = x[1:] - x[:-1]
    # cyclicly repeat `x` at the beginning/end
    dx0 = (x[0] - x[-1] if cyclic else 0)[None]
    # handle NaN's
    indnan = torch.where(torch.isnan(x))[0]
    if indnan.size(0):
        x[indnan] = float("inf")
        dx[torch.where(torch.isnan(dx))[0]] = float("inf")

    # index of None/raising/falling edges
    ine, ire, ife = torch.as_tensor([[], [], []], dtype=int)
    if edge is not None:
        ine = torch.where(
            (torch.cat((dx, dx0)) < 0) & (torch.cat((dx0, dx)) > 0)
        )[0]
    else:
        if edge in ("rising", "both"):
            ire = torch.where(
                (torch.cat((dx, dx0)) <= 0) & (torch.cat((dx0, dx)) > 0)
            )[0]
        if edge in ("falling", "both"):
            ife = torch.where(
                (torch.cat((dx, dx0)) < 0) & (torch.cat((dx0, dx)) >= 0)
            )[0]
    ind = torch.unique(torch.cat((ine, ire, ife)))
    # handle NaN's
    if ind.size(0) and indnan.size(0):
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[
            in1D(
                ind,
                torch.unique(torch.cat((indnan, indnan - 1, indnan + 1))),
                invert=True,
            )
        ]
    if not cyclic:
        # first and last values of x cannot be peaks
        if ind.size(0) and ind[0] == 0:
            ind = ind[1:]
        if ind.size(0) and ind[-1] == x.size(0) - 1:
            ind = ind[:-1]
    # remove peaks < minimum peak height
    # or remove valleys > maximum peak height
    if ind.size(0) and mph is not None:
        ind = ind[x[ind] >= mph] if valley else ind[x[ind] <= mph]
    # remove peaks - neighbors < threshold
    # or remove valleys - neighbors > threshold
    if ind.size(0) and threshold > 0:
        if valley:
            # dx = torch.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]).max(dim=0)
            dx = torch.cat([(x[ind] - x[ind - 1])[None, :], (x[ind] - x[ind + 1])[None, :]], dim=0).max(dim=0)
            ind = torch.index_select(ind, 0, torch.where(dx <= threshold)[0])
        else:
            # dx = torch.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]).min(dim=0)
            dx = torch.cat([(x[ind] - x[ind - 1])[None, :], (x[ind] - x[ind + 1])[None, :]], dim=0).min(dim=0)
            ind = torch.index_select(ind, 0, torch.where(dx > threshold)[0])

    if not ind.size(0):
        return torch.as_tensor([], dtype=int)
    # detect small peaks closer than minimum peak distance
    if mpd > 1:
        ind = ind[torch.argsort(x[ind])].flip(0)  # sort ind by peak height
        idel = torch.zeros_like(ind, dtype=bool)
        for i in range(ind.size(0)):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (
                    x[ind[i]] > x[ind] if kpsh else True
                )
                idel[i] = 0  # Keep current peak
        # remove the small peaks
        ind = ind[~idel]
        if sort == "index":
            # sort indexes by their occurrence
            ind = torch.sort(ind)
    elif sort == "value":
        # sort indexes by their peak height
        ind = ind[torch.argsort(x[ind])].flip(0)

    return ind


class Frontend(nn.Module):
    def __init__(self,
                 idim: int,
                 use_vad_mask: bool = False,
                 randomly_bypass_frontend: bool = False,
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
        assert bnmask in (3, 6), bnmask

        self.use_vad_mask = use_vad_mask
        if not self.use_vad_mask:
            from espnet.nets.pytorch_backend.frontends.mask_estimator_v2 import MaskEstimator
            logging.warning('Using normal T-F masks')
        else:
            from espnet.nets.pytorch_backend.frontends.mask_estimator_vad import MaskEstimator
            logging.warning('Using VAD-like masks (same value for all frequencies in each frame)')

        self.randomly_bypass_frontend = randomly_bypass_frontend
        if randomly_bypass_frontend:
            logging.warning('Randomly bypassing the frontend for single-speaker data')

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

        self.ref_channel = ref_channel
        logging.warning('Ref channel is {}'.format(ref_channel))
        if self.use_beamformer:
            self.ref = AttentionReference(idim, badim) if ref_channel < 0 else None
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

                if getattr(self, "randomly_bypass_frontend", False) and self.nmask == 3:
                    # single-speaker
                    choices.append((False, False))

                use_wpe, use_beamformer = \
                    choices[numpy.random.randint(len(choices))]

            else:
                use_wpe = self.use_wpe and self.taps > 0
                use_beamformer = self.use_beamformer

            if masks is not None:
                if len(masks) == 6:
                    # 2-speaker
                    wpe_masks = masks[:2]
                    beamforming_masks = masks[2:]
                elif len(masks) == 3:
                    # single-speaker
                    wpe_masks = masks[:1]
                    beamforming_masks = masks[1:]
                else:
                    raise ValueError("Invalid length of masks: %d" % len(masks))
                mask = masks
            else:
                data = h.permute(0, 3, 2, 1)
                mask, _ = self.mask(data.float(), ilens)
                if len(mask) == 6:
                    # 2-speaker
                    wpe_masks = mask[:2]
                    beamforming_masks = mask[2:]
                elif len(mask) == 3:
                    # single-speaker
                    wpe_masks = mask[:1]
                    beamforming_masks = mask[1:]
                else:
                    raise ValueError("Invalid length of masks: %d" % len(mask))

            if not use_beamformer and not use_wpe:
                print(">", end="", flush=True)
                # randomly select one channel as output for each sample in the minibatch
                idx = torch.randint(low=0, high=h.shape[2], size=(h.shape[0], 1, 1, 1), device=h.device)
                h = h.gather(2, idx.expand(-1, h.shape[1], 1, h.shape[3])).squeeze(2)

            elif self.use_beamforming_first and use_beamformer:
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
                    if isinstance(h, list) and len(h) == 1:
                        h, _ = self.beamforming(h[0], ilens, power=powers, irms=beamforming_masks)
                    elif isinstance(h, list):
                        for i, hspk in enumerate(h):
                            h[i], _ = self.beamforming(hspk, ilens, power=powers[i], irms=beamforming_masks[i::2])
                    else:
                        h, _ = self.beamforming(h, ilens, power=powers, irms=beamforming_masks)

            mask = [m.transpose(-1, -3) for m in mask]

        if isinstance(h, list):
            h = [hh.float() for hh in h] if len(h) > 1 else h[0].float()
        else:
            h = h.float()
        return h, ilens, mask

    def get_steering_vector(
        self,
        relative_mic_angle: torch.Tensor,
        relative_mic_dist: torch.Tensor,
        doa: torch.Tensor,
        freq: torch.Tensor,
        inverse: bool = False,
        sound_velocity: float = 343,
    ) -> torch.Tensor:
        """Return the normalized steering vector given the array geometry and DOA.

        Args:
            relative_mic_angle (torch.Tensor): relative angular distances (in degrees)
                                            between each mic and ref mic
            relative_mic_dist (torch.Tensor): relative distances (in meters) between
                                            each mic and array center
            doa (torch.Tensor): a list of direction of arrivals in degrees (-180, 180]
            freq (torch.Tensor): frequency bins in Hz
            inverse (bool): True to inverse the phase shift
            sound_velocity (float): sound velocity in meters/second
        Returns:
            sv (ComplexTensor): list of steering vectors corresponding to the list of doas
                    All steering vectors are normalized so that the `ref_mic`-th
                    element of each steering vector is 1. (num_doa, num_freq, num_mic)
        """
        PI = numpy.pi
        # (num_doa, num_mic)
        delta_ang = doa.unsqueeze(dim=1) - relative_mic_angle.unsqueeze(dim=0)
        # relative time delay in seconds
        delay = relative_mic_dist * torch.cos(delta_ang / 180.0 * PI) / sound_velocity
        # (num_doa, num_freq, num_mic)
        signed_2pi = 2 * PI if inverse else -2 * PI
        phase_shift = torch.einsum("f,dc->dfc", freq, signed_2pi * delay)
        # sv = exp(j 2pi f delay)
        sv = ComplexTensor(torch.cos(phase_shift), torch.sin(phase_shift))
        return sv

    def _weighted_srp_phat(self, signal, sv, doa, mask=None):
        """Steered Response Power PHAse Transform (SRP-PHAT)"""
        # sv difference between i- and j-th channels (num_doa, F, C, C)
        # only take the elements above the 1st diagnoal for removing redundancy, e.g.:
        #  | 0  x  x  x |
        #  | 0  0  x  x |
        #  | 0  0  0  x |
        #  | 0  0  0  0 |
        sv_ij = FC.einsum("dfi,dfj->dfij", sv, sv.conj())
        sv_ij = ComplexTensor(
            torch.triu(sv_ij.real, diagonal=1),
            torch.triu(sv_ij.imag, diagonal=1),
        )

        phase = signal / signal.abs().clamp_min(1e-10)
        # IPD between j- and i-th channels (F, C, C)
        if mask is not None:
            cc_ji = FC.einsum("tfj,tfi->fji", phase * mask.unsqueeze(-1), phase.conj()) / phase.size(0)
        else:
            cc_ji = FC.einsum("tfj,tfi->fji", phase, phase.conj()) / phase.size(0)
        cc_ji = ComplexTensor(
            torch.triu(cc_ji.real, diagonal=1),
            torch.triu(cc_ji.imag, diagonal=1),
        )

        # cross-correlation (num_doa, F, C, C) -> (num_doa,)
        # This is equivalent to `np.real(sv_ij[:, None, ...] * cc_ji[None, ...])`
        # but avoids computing the imaginary part that we end up discarding
        R = torch.einsum("dfij,fij->d", sv_ij.real, cc_ji.real) - torch.einsum(
            "dfij,fij->d", sv_ij.imag, cc_ji.imag
        )
        peak_indices = find_peaks(R, cyclic=True, edge="both", sort="value")
        if len(peak_indices) == 0:
            return None
        doa_hat = doa[peak_indices[0]]
        return doa_hat

    def _resolve_frequency_permutation(
        self,
        x: ComplexTensor,
        enhanced: List[ComplexTensor],
        sensor_pos: List[List],
        fs: int = 16000,
        freq_min: float = 400,
        freq_max: float = 4000,
        resolution: float = 1.0,
        sound_velocity: float = 343,
        threshold: float = 180.0,
    ):
        # Mitigate the frequency permutation problem via DOA estimation
        if isinstance(enhanced, ComplexTensor) or len(enhanced) <= 1:
            return enhanced

        # (B=1, T, C, F)
        if x.dim() != 4:
            raise ValueError(f'Input dim must be 4: {x.dim()}')
        assert x.size(0) == enhanced[0].size(0) == 1, (x.shape, enhanced[0].shape)

        ref_mic = 0
        # sensor_pos = [
        #    (3.957, 3.083, 1.517),  # (x, y, z)
        #    (4.02, 3.161, 1.517),
        #    (3.984, 3.254, 1.519),
        #    (3.885, 3.27, 1.521),
        #    (3.822, 3.192, 1.521),
        #    (3.858, 3.099, 1.519),
        #]
        mics = torch.stack([torch.as_tensor(mic) for mic in sensor_pos], dim=0)
        array_center = mics.mean(dim=0, keepdim=True)
        relative_mics = mics - array_center
        relative_mic_pos = FC.stack([ComplexTensor(*p[:2]) for p in relative_mics], dim=0)
        # relative angle between mic_i and mic_ref
        # relative_mic_angle = relative_mic_pos.angle().rad2deg()
        M_180_PI = 57.295779513082320876798154814105170332405472466564
        relative_mic_angle = relative_mic_pos.angle() * M_180_PI
        # normalize the rotation so that the DOA of ref_mic w.r.t. array center is 0
        mic_ref_rotation = relative_mic_angle[ref_mic]
        relative_mic_angle = relative_mic_angle - mic_ref_rotation
        relative_mic_dist = relative_mic_pos.abs()

        # mask the multi-channel input signal based on the beamformed signals
        masks = [
            # (T, F)
            (enh[0].abs() / x[0, :, self.ref_channel].abs().clamp(min=1.0e-08)).clamp(
                min=1.0e-08, max=1.0
            )
            for enh in enhanced
        ]

        # (T, C, F) -> (T, F, C)
        signal = x[0].permute(0, 2, 1)
        freq = torch.linspace(0, fs // 2, signal.shape[1], dtype=signal.real.dtype)
        freq = freq.to(device=signal.device)

        num_doa = int(360 / resolution)
        # (-180, 180]
        doa = 180 - torch.arange(0, num_doa, dtype=freq.dtype).flip(0) * resolution
        doa = doa.to(device=signal.device)
        # (num_doa, F, C)
        sv = self.get_steering_vector(
            relative_mic_angle,
            relative_mic_dist,
            doa,
            freq,
            inverse=True,
            sound_velocity=sound_velocity,
        ).to(device=signal.device)

        ffrom = torch.where(freq >= freq_min)[0][0]
        fto = torch.where(freq <= freq_max)[0][-1] + 1

        # obtain DOAs for all speakers based on a range of frequencies
        doas = [
            self._weighted_srp_phat(
                signal[:, ffrom:fto], sv[:, ffrom:fto], doa, mask=mask[:, ffrom:fto]
            )
            for mask in masks
        ]
        if any([ang is None for ang in doas]):
            print("Skipping due to the failure to find all DOAs")
            return enhanced
        doas = torch.as_tensor(doas)
        all_permutations = list(permutations(range(len(doas))))

        def pair_loss(permutation, ref, inf):
            return sum(
                [
                    ((ref[s] - inf[t] + 180) % 360 - 180).abs()
                    for s, t in enumerate(permutation)
                ]
            )

        # (B=1, T, F, num_spk)
        enhanced = FC.stack(enhanced, dim=-1)
        # test DOA deviation for individual frequency bins
        for f in range(signal.shape[1]):
            doas_hat = [
                self._weighted_srp_phat(
                    signal[:, f:f+1], sv[:, f:f+1], doa, mask=mask[:, f:f+1]
                )
                for mask in masks
            ]
            if any([ang is None for ang in doas_hat]):
                continue
            doas_hat = torch.as_tensor(doas_hat)
            losses = torch.stack(
                [pair_loss(p, doas, doas_hat) for p in all_permutations], dim=0
            )
            loss, perm_ = torch.min(losses, dim=0)
            if loss > threshold:
                continue
            perm = torch.tensor(
                all_permutations[perm_], device=signal.device, dtype=torch.long
            )
            # exchange the frequency bin across the separated spectra
            enhanced[:, :, f] = enhanced[:, :, f, perm]

        return enhanced.unbind(-1)


def frontend_for(args, idim):
    return Frontend(
        idim=idim,
        use_vad_mask=getattr(args, "use_vad_mask", False),
        randomly_bypass_frontend=getattr(args, "randomly_bypass_frontend", False),
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
