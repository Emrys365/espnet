"""
This script is used to construct End-to-End models of multi-speaker ASR with transformers.

Copyright 2019 Shigeki Karita
          2019 Xuankai Chang
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from argparse import Namespace
from distutils.util import strtobool
from distutils.version import LooseVersion
from itertools import permutations
import logging
import math
import yaml

import chainer
import ci_sdr
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr_mix import PIT
#from espnet.nets.pytorch_backend.e2e_asr_mix import Reporter
from espnet.nets.pytorch_backend.e2e_asr_mix import E2E as E2E_ASR_Mix
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2E_ASR
from espnet.nets.pytorch_backend.frontends.feature_transform import feature_transform_for
from espnet.nets.pytorch_backend.frontends.frontend_v2_complex_mask import _compress_mask
from espnet.nets.pytorch_backend.frontends.stft import Stft
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer


is_torch_1_3_plus = LooseVersion(torch.__version__) >= LooseVersion("1.3.0")
EPS = torch.finfo(torch.get_default_dtype()).eps


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss_enh, loss_ctc, loss_att, acc, cer, wer, mtl_loss):
        """Define reporter."""
        chainer.reporter.report({"loss_enh": loss_enh}, self)
        chainer.reporter.report({"loss_ctc": loss_ctc}, self)
        chainer.reporter.report({"loss_att": loss_att}, self)
        chainer.reporter.report({"acc": acc}, self)
        chainer.reporter.report({"cer": cer}, self)
        chainer.reporter.report({"wer": wer}, self)
        logging.info("mtl loss:" + str(mtl_loss))
        chainer.reporter.report({"loss": mtl_loss}, self)


def _create_mask_label(mix_spec, ref_spec, mask_type="IAM"):
        """Create mask label.

        Args:
            mix_spec: ComplexTensor(B, T, F)
            ref_spec: List[ComplexTensor(B, T, F), ...]
            mask_type: str
        Returns:
            labels: List[Tensor(B, T, F), ...] or List[ComplexTensor(B, T, F), ...]
        """

        # Must be upper case
        assert mask_type in [
            "IBM", "IRM", "cIRM", "IAM", "IAM^2", "VAD", "VAD^2", "PSM", "NPSM", "PSM^2"
        ], f"mask type {mask_type} not supported"

        def _power_spec(spec):
            if isinstance(spec, ComplexTensor) or (
                hasattr(spec, "real") and hasattr(spec, "imag")
            ):
                return spec.real ** 2 + spec.imag ** 2
            else:
                return spec ** 2

        mask_label = []
        for i in range(len(ref_spec)):
            r = (
                ref_spec[i]
                if ref_spec[i].dim() == mix_spec.dim()
                else ref_spec[i].unsqueeze(-2)
            )
            mask = None
            if mask_type == "cIRM":
                # Reference: Complex Ratio Masking for Monaural Speech
                # Separation; Williamson et al, 2016
                denom = mix_spec.real.pow(2) + mix_spec.imag.pow(2) + EPS
                mask_real = (mix_spec.real * r.real + mix_spec.imag * r.imag) / denom
                mask_imag = (mix_spec.real * r.imag - mix_spec.imag * r.real) / denom
                # Compress with the hyperbolic tangent
                mask = ComplexTensor(
                    ESPnetEnhancementModel._compress_mask(mask_real),
                    ESPnetEnhancementModel._compress_mask(mask_imag),
                )
            elif mask_type == "IAM":
                mask = abs(r) / (abs(mix_spec) + EPS)
                mask = mask.clamp(min=0, max=1)
            elif mask_type == "IAM^2":
                mask = _power_spec(r) / (_power_spec(mix_spec) + EPS)
                mask = mask.clamp(min=0, max=1)
            elif mask_type == "VAD":
                mask = abs(r) / (abs(mix_spec) + EPS)
                mask = (
                    mask.clamp(min=0, max=1)
                    .mean(dim=-1, keepdims=True)
                    .expand(mask.shape)
                )
            elif mask_type == "VAD^2":
                mask = _power_spec(r) / (_power_spec(mix_spec) + EPS)
                mask = (
                    mask.clamp(min=0, max=1)
                    .mean(dim=-1, keepdims=True)
                    .pow(2)
                    .expand(mask.shape)
                )
            elif mask_type == "PSM" or mask_type == "NPSM":
                complex_eps = r.real.new_full((), EPS)
                complex_eps = ComplexTensor(complex_eps, complex_eps)
                phase_r = r / (abs(r) + complex_eps)
                phase_mix = mix_spec / (abs(mix_spec) + complex_eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                    phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (abs(r) / (abs(mix_spec) + EPS)) * cos_theta
                mask = (
                    mask.clamp(min=0, max=1)
                    if mask_type == "NPSM"
                    else mask.clamp(min=-1, max=1)
                )
            elif mask_type == "PSM^2":
                complex_eps = r.real.new_full((), EPS)
                complex_eps = ComplexTensor(complex_eps, complex_eps)
                # This is for training beamforming masks
                phase_r = r / abs(r + complex_eps)
                phase_mix = mix_spec / abs(mix_spec + complex_eps)
                # cos(a - b) = cos(a)*cos(b) + sin(a)*sin(b)
                cos_theta = (
                    phase_r.real * phase_mix.real + phase_r.imag * phase_mix.imag
                )
                mask = (_power_spec(r) / (_power_spec(mix_spec) + EPS)) * cos_theta
                mask = mask.clamp(min=-1, max=1)
            assert mask is not None, f"mask type {mask_type} not supported"
            mask_label.append(mask)
        return mask_label


def compute_enh_loss(
    speech_mix, speech_lengths, speech_ref, speech_est, mask_est, loss_type, mask_type="cIRM", stft=None, ref_channel=0
):
    """Compute the frontend loss.

    Args:
        speech_mix (ComplexTensor): input mixture spectrum of shape: (Batch, T, C, F) or (Batch, T, F)
        speech_lengths (torch.Tensor): original input waveform lengths of shape: (Batch,)
        speech_ref (ComplexTensor): clean speech spectrum of shape: (num_spk, Batch, T, C, F) or (num_spk, Batch, T, F)
        speech_ref (ComplexTensor): clean speech waveform of shape: (num_spk, Batch, T', C) or (num_spk, Batch, T')
        speech_est (List[ComplexTensor]): estimated clean speech spectrum of shape: num_spk * (Batch, T, F)
        mask_est (List[ComplexTensor]): estimated masks of shape: 6 * (Batch, T, C, F) or 6 * (Batch, T, F)
        loss_type (str): one of "mask_mse", "spectrum", "snr", "si_snr", "ci_sdr"
        mask_type (str): one of "IBM", "IRM", "cIRM", "IAM", "IAM^2", "VAD", "VAD^2", "PSM", "NPSM", "PSM^2"
        stft (callable): instance of the Stft module
        ref_channel (int): reference channel, default is 0
    Returns:
        loss: (Batch,)
        min_perm: (Batch, num_spk) best permutation for speech_est
    """
    if loss_type == "mask_mse":
        mask_ref = _create_mask_label(speech_mix, speech_ref, mask_type=mask_type)
        loss, perm = _permutation_loss(mask_ref, mask_est, tf_mse_loss)
    elif loss_type == "spectrum":
        if not isinstance(speech_ref, ComplexTensor):
            spectrum_ref = [stft(sr, speech_lengths)[0] for sr in speech_ref]
        loss, perm = _permutation_loss(spectrum_ref, spectrum_pre, tf_mse_loss)
    elif loss_type in ("snr", "si_snr", "ci_sdr"):
        assert stft is not None
        if loss_type == "snr":
            loss_func = snr_loss
        elif loss_type == "si_snr":
            loss_func = si_snr_loss
        elif loss_type == "ci_sdr":
            loss_func = ci_sdr_loss
        else:
            raise ValueError("Unknown loss_type: %s" % loss_type)

        speech_est_t = [stft.inverse(ps, speech_lengths)[0] for ps in speech_est]
        assert speech_est_t[0].dim() == 2, speech_est_t[0].dim()

        if isinstance(speech_ref, ComplexTensor):
            if speech_ref.dim() == 5:
                speech_ref = speech_ref[..., ref_channel, :]
            speech_ref = [stft.inverse(ps, speech_lengths)[0] for ps in speech_ref]
        else:
            # raw waveform
            if speech_ref.dim() == 4:
                speech_ref = speech_ref[..., ref_channel]

        loss, perm = _permutation_loss(speech_ref, speech_est_t, loss_func)
    else:
        raise ValueError("Unknown loss_type: %s" % loss_type)

    return loss, perm


def tf_mse_loss(ref, inf):
        """time-frequency MSE loss.

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        if not is_torch_1_3_plus:
            # in case of binary masks
            ref = ref.type(inf.dtype)
        diff = ref - inf
        if isinstance(diff, ComplexTensor):
            mseloss = diff.real ** 2 + diff.imag ** 2
        else:
            mseloss = diff ** 2
        if ref.dim() == 3:
            mseloss = mseloss.mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = mseloss.mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )

        return mseloss


def tf_log_mse_loss(ref, inf):
        """time-frequency log-MSE loss.

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        if not is_torch_1_3_plus:
            # in case of binary masks
            ref = ref.type(inf.dtype)
        diff = ref - inf
        if isinstance(diff, ComplexTensor):
            log_mse_loss = diff.real ** 2 + diff.imag ** 2
        else:
            log_mse_loss = diff ** 2
        if ref.dim() == 3:
            log_mse_loss = torch.log10(log_mse_loss.sum(dim=[1, 2])) * 10
        elif ref.dim() == 4:
            log_mse_loss = torch.log10(log_mse_loss.sum(dim=[1, 2, 3])) * 10
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )

        return log_mse_loss


def tf_l1_loss(ref, inf):
        """time-frequency L1 loss.

        Args:
            ref: (Batch, T, F) or (Batch, T, C, F)
            inf: (Batch, T, F) or (Batch, T, C, F)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        if not is_torch_1_3_plus:
            # in case of binary masks
            ref = ref.type(inf.dtype)
        if isinstance(inf, ComplexTensor):
            l1loss = abs(ref - inf + EPS)
        else:
            l1loss = abs(ref - inf)
        if ref.dim() == 3:
            l1loss = l1loss.mean(dim=[1, 2])
        elif ref.dim() == 4:
            l1loss = l1loss.mean(dim=[1, 2, 3])
        else:
            raise ValueError(
                "Invalid input shape: ref={}, inf={}".format(ref.shape, inf.shape)
            )
        return l1loss


def snr_loss(ref, inf):
    """SNR loss

    Args:
        ref: (Batch, samples)
        inf: (Batch, samples)
    Returns:
        loss: (Batch,)
    """
    noise = inf - ref

    snr = 20 * (
        torch.log10(torch.norm(ref, p=2, dim=1).clamp(min=EPS))
        - torch.log10(torch.norm(noise, p=2, dim=1).clamp(min=EPS))
    )
    return -snr


def si_snr_loss(ref, inf):
        """SI-SNR loss

        Args:
            ref: (Batch, samples)
            inf: (Batch, samples)
        Returns:
            loss: (Batch,)
        """
        ref = ref / torch.norm(ref, p=2, dim=1, keepdim=True)
        inf = inf / torch.norm(inf, p=2, dim=1, keepdim=True)

        s_target = (ref * inf).sum(dim=1, keepdims=True) * ref
        e_noise = inf - s_target

        si_snr = 20 * (
            torch.log10(torch.norm(s_target, p=2, dim=1).clamp(min=EPS))
            - torch.log10(torch.norm(e_noise, p=2, dim=1).clamp(min=EPS))
        )
        return -si_snr


def si_snr_loss_zeromean(ref, inf):
        """SI-SNR loss with zero-mean in pre-processing.

        Args:
            ref: (Batch, samples)
            inf: (Batch, samples)
        Returns:
            loss: (Batch,)
        """
        assert ref.size() == inf.size()
        B, T = ref.size()
        # mask padding position along T

        # Step 1. Zero-mean norm
        mean_target = torch.sum(ref, dim=1, keepdim=True) / T
        mean_estimate = torch.sum(inf, dim=1, keepdim=True) / T
        zero_mean_target = ref - mean_target
        zero_mean_estimate = inf - mean_estimate

        # Step 2. SI-SNR with order
        # reshape to use broadcast
        s_target = zero_mean_target  # [B, T]
        s_estimate = zero_mean_estimate  # [B, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=1, keepdim=True)  # [B, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=1, keepdim=True) + EPS  # [B, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, T]

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + EPS
        )
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B]

        return -1 * pair_wise_si_snr


def ci_sdr_loss(ref, inf):
        """CI-SDR loss

        Reference:
            Convolutive Transfer Function Invariant SDR Training Criteria for
            Multi-Channel Reverberant Speech Separation; C. Boeddeker et al., 2021;
            https://arxiv.org/abs/2011.15003

        Args:
            ref: (Batch, samples)
            inf: (Batch, samples)
        Returns:
            loss: (Batch,)
        """
        assert ref.shape == inf.shape, (ref.shape, inf.shape)
        return ci_sdr.pt.ci_sdr_loss(inf, ref, compute_permutation=False)


def _permutation_loss(ref, inf, criterion, perm=None):
        """The basic permutation loss function.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...] x n_spk
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm (torch.Tensor): specified permutation (batch, num_spk)
        Returns:
            loss (torch.Tensor): minimum loss with the best permutation (batch)
            perm (torch.Tensor): permutation for inf (batch, num_spk)
                                 e.g. tensor([[1, 0, 2], [0, 1, 2]])
        """
        assert len(ref) == len(inf), (len(ref), len(inf))
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t]) for s, t in enumerate(permutation)]
            ) / len(permutation)

        if perm is None:
            device = ref[0].device
            all_permutations = list(permutations(range(num_spk)))
            losses = torch.stack([pair_loss(p) for p in all_permutations], dim=1)
            loss, perm = torch.min(losses, dim=1)
            perm = torch.index_select(
                torch.tensor(all_permutations, device=device, dtype=torch.long),
                0,
                perm,
            )
        else:
            loss = torch.tensor(
                [
                    torch.tensor(
                        [
                            criterion(
                                ref[s][batch].unsqueeze(0), inf[t][batch].unsqueeze(0)
                            )
                            for s, t in enumerate(p)
                        ]
                    ).mean()
                    for batch, p in enumerate(perm)
                ]
            )

        return loss.mean(), perm



class E2E(E2E_ASR, ASRInterface, torch.nn.Module):
    @staticmethod
    def add_arguments(parser):
        E2E_ASR.add_arguments(parser)
        E2E_ASR_Mix.encoder_mix_add_arguments(parser)
        E2E.initialize_add_arguments(parser)
        E2E.beamformer_add_arguments(parser)
        return parser

    @staticmethod
    def initialize_add_arguments(parser):
        """Add arguments for initializing multi-speaker model."""
        group = parser.add_argument_group("E2E multichannel model initialization for multi-speaker")
        group.add_argument('--init-asr', default='', nargs='?',
                           help='Initialze the asr model from')
        group.add_argument('--init-frontend', default='', nargs='?',
                           help='Initialze the frontend model from')
        group.add_argument('--init-from-mdl', default='', nargs='?',
                           help='Initialze from another model')
        return parser

    @staticmethod
    def beamformer_add_arguments(parser):
        """Add arguments for multi-speaker beamformer."""
        group = parser.add_argument_group("E2E transformer-based beamformer setting for multi-speaker")
        # transformer-based beamformer
        group.add_argument('--beamformer-time-restricted-window', default=15, type=int,
                           help='Context window for time restricted self-attention in transformer-based neural beamformer.')
        # rnn-based beamfomer
        group.add_argument('--beamformer-type', type=str, default="mvdr", choices=["mvdr", "mpdr", "wmpdr", 'mvdr_souden', 'mpdr_souden', 'wmpdr_souden', 'wpd_souden', 'wpd'],
                           help='which beamforming implementation to be used')
        group.add_argument('--use-beamforming-first', type=strtobool, default=False,
                           help='whether to perform beamforming before WPE')
        group.add_argument('--use-WPD-frontend', type=strtobool, default=False,
                           help='use WPD frontend instead of WPE + MVDR beamformer')
        group.add_argument('--wpd-opt', type=float, default=1, choices=[1, 2, 3, 4, 5, 5.2, 6],
                           help='which WPD implementation to be used')
        group.add_argument('--use-padertorch-frontend', type=strtobool, default=False,
                           help='use padertorch-like frontend')
        group.add_argument('--use-complex-mask', type=strtobool, default=False,
                           help='use complex masks instead of magnitude masks, only works when use_padertorch_frontend is True')
        group.add_argument('--use-vad-mask', type=strtobool, default=False,
                           help='use VAD-like masks instead of T-F masks, only works when use_padertorch_frontend is True and use_complex_mask is False')
        group.add_argument('--multich-epochs', default=-1, type=int,
                           help='From which epoch the multichannel data is used')
        group.add_argument('--test-btaps', type=int, default=-1,
                           help='use the specified btaps for testing')
        group.add_argument('--test-nmics', type=int, default=-1,
                           help='use the specified number of microphones for testing')
        group.add_argument('--atf-iterations', type=int, default=2,
                           help='use the specified number of iterations for estimating the steering vector, only used with MVDR/MPDR/wMPDR beamformers')
        group.add_argument('--wpe-tag', type=str, default='default',
                           help='WPE tag for selecting a specific implementation')
        group.add_argument('--beamforming-tag', type=str, default='default',
                           help='WPE tag for selecting a specific implementation')
        group.add_argument('--bp-enh-loss', type=strtobool, default=False,
                           help='Whether to back-propagate loss_enh')
        group.add_argument('--enh-loss-type', type=str, default="mask_mse", choices=["mask_mse", "spectrum", "snr", "si_snr", "ci_sdr"],
                           help='Type of enh loss')
        group.add_argument('--mask-type', type=str, default="cIRM",
                           choices=["IBM", "IRM", "cIRM", "IAM", "IAM^2", "VAD", "VAD^2", "PSM", "NPSM", "PSM^2"], help='Type of enh loss')
        group.add_argument('--enh-loss-weight', type=float, default=1.0,
                           help='weight for enh loss')
        return parser

    @property
    def attention_plot_class(self):
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        torch.nn.Module.__init__(self)
        if getattr(args, "use_frontend", False):  # use getattr to keep compatibility
            if getattr(args, "use_WPD_frontend", False):
                from espnet.nets.pytorch_backend.frontends.frontend_wpd_v5 import frontend_for
                logging.warning('Using WPD frontend')
            elif getattr(args, "use_padertorch_frontend", False):
                if getattr(args, "use_complex_mask", False):
                    from espnet.nets.pytorch_backend.frontends.frontend_v2_complex_mask import frontend_for
                    logging.warning('Using padertorch-like frontend with complex masks')
                else:
                    from espnet.nets.pytorch_backend.frontends.frontend_v2 import frontend_for
                    logging.warning('Using padertorch-like frontend')
            else:
                from espnet.nets.pytorch_backend.frontends.frontend import frontend_for

            self.frontend = frontend_for(args, idim)
            self.feature_transform = feature_transform_for(args, (idim - 1) * 2)
            idim = args.n_mels + 3 if getattr(args, "fbank_pitch", None) is not None else args.n_mels
        else:
            self.frontend = None
        
        with open(args.preprocess_conf) as file:
            preproc_conf = yaml.load(file, Loader = yaml.FullLoader)
            preproc_conf = preproc_conf['process'][0]
        self.stft = Stft(
            win_length=preproc_conf['win_length'],
            n_fft=preproc_conf['n_fft'],
            hop_length=preproc_conf['n_shift'],
            window=preproc_conf['window'],
        )

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        self.decoder = Decoder(
            odim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        #self.subsample = [1]
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.reporter = Reporter()
        self.num_spkrs = args.num_spkrs
        self.spa = args.spa
        self.pit = PIT(self.num_spkrs)
        self.bp_enh_loss = getattr(args, 'bp_enh_loss', False)
        self.enh_loss_type = getattr(args, "enh_loss_type", "mask_mse")
        self.mask_type= getattr(args, "mask_type", "cIRM")
        self.ref_channel = args.ref_channel

        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )
        # self.verbose = args.verbose
        self.reset_parameters(args)
        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=False, ignore_nan_grad=args.ignore_nan_grad)
        else:
            self.ctc = None

        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None
        self.enh_loss_weight = args.enh_loss_weight

    def reset_parameters(self, args):
        # initialize parameters
        initialize(self, args.transformer_init)

    def add_sos_eos(self, ys_pad):
        from espnet.nets.pytorch_backend.nets_utils import pad_list
        eos = ys_pad.new([self.eos])
        sos = ys_pad.new([self.sos])
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        return pad_list(ys_in, self.eos), pad_list(ys_out, self.ignore_id)

    def target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != self.ignore_id
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, xs_pad, ilens, ys_pad, tgt_pad=None, wav_lens=None):
        '''E2E forward

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :param torch.Tensor tgt_pad: batch of padded reference signals (B, Tmax, idim)
        :param torch.Tensor wav_lens: batch of lengths of original waveform sequences (B)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        '''
        # 0. forward frontend
        if self.frontend is not None:
            # train frontend on CPU to make it more stable
            xs_pad = to_device(self, to_torch_tensor(xs_pad))
#            np.save('/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_multich/dump_tmp/xs_pad_{}.npy'.format(str(xs_pad.device)).replace(':', ''), xs_pad.detach().cpu().numpy())
            hs_pad, hlens, mask = self.frontend(xs_pad, ilens)
            if isinstance(hs_pad, (list, tuple)):

#                np.save('/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_multich/dump_tmp/hs_pad_{}.npy'.format(str(hs_pad[0].device)).replace(':', ''), [h.detach().cpu().numpy() for h in hs_pad])
#                np.save('/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_multich/dump_tmp/mask_{}.npy'.format(str(mask[0].device)).replace(':', ''), [m.detach().cpu().numpy() for m in mask])
                if tgt_pad is not None:
                    tgt_pad = to_device(self, to_torch_tensor(tgt_pad))
                    #with torch.no_grad():
                    #    loss_enh_perm = torch.stack(
                    #        [
                    #            tf_mse_loss(
                    #                hs_pad[i // self.num_spkrs],
                    #                tgt_pad[i % self.num_spkrs]
                    #            )
                    #            for i in range(self.num_spkrs ** 2)
                    #        ],
                    #    dim=1)  # (B, num_spkrs^2)
                    #loss_enh, min_perm = self.pit.pit_process(loss_enh_perm)
                    loss_enh, min_perm = compute_enh_loss(
                        xs_pad, wav_lens, tgt_pad, hs_pad, mask, self.enh_loss_type,
                        mask_type=self.mask_type, stft=self.stft, ref_channel=self.ref_channel,
                    )
                    # scale by input length
                    #loss_enh = loss_enh * hs_pad[0].shape[1]
                else:
                    loss_enh, min_perm = None, None

                hlens_n = [None] * self.num_spkrs
                for i in range(self.num_spkrs):
                    hs_pad[i], hlens_n[i] = self.feature_transform(hs_pad[i].float(), hlens)
                hlens = hlens_n
            else:
                # move to GPU for ASR
                hs_pad, hlens = self.feature_transform(hs_pad.float(), hlens)
                loss_enh, min_perm = None, None
        else:
            hs_pad, hlens = xs_pad.float(), ilens
            loss_enh, min_perm = None, None

        # 1. forward encoder
        if not isinstance(hs_pad, (list, tuple)):  # single-channel input xs_pad (single-speaker)
            hs_pad = hs_pad[:, :max(hlens)]  # for data parallel
            src_mask = (~make_pad_mask(hlens.tolist())).to(hs_pad.device).unsqueeze(-2)
            hs_pad, hs_mask = self.encoder(hs_pad, src_mask)
        else:  # multi-channel multi-speaker input xs_pad
            src_mask = [None] * self.num_spkrs
            hs_mask = [None] * self.num_spkrs
            for i in range(self.num_spkrs):
                hs_pad[i] = hs_pad[i][:, :max(hlens[i])]  # for data parallel
                src_mask[i] = (~make_pad_mask(hlens[i].tolist())).to(hs_pad[i].device).unsqueeze(-2)
                hs_pad[i], hs_mask[i] = self.encoder(hs_pad[i], src_mask[i])
        #self.hs_pad = hs_pad

        # CTC loss
        assert self.mtlalpha > 0.0 and self.mtlalpha < 1.0
        batch_size = len(ilens)  # hs_pad.size(0)
        if not isinstance(hs_pad, (list, tuple)):  # single-speaker input xs_pad
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = torch.mean(self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad))
        else:  # multi-speaker input xs_pad
            ys_pad = ys_pad.transpose(0, 1)  # (num_spkrs, B, Lmax)
            hs_len = [hs_mask[i].view(batch_size, -1).sum(1) for i in range(self.num_spkrs)]
            if min_perm is None:
                loss_ctc_perm = torch.stack([
                    self.ctc(
                        hs_pad[i // self.num_spkrs].view(batch_size, -1, self.adim),
                        hs_len[i // self.num_spkrs],
                        ys_pad[i % self.num_spkrs],
                    )
                    for i in range(self.num_spkrs ** 2)
                ], dim=1)  # (B, num_spkrs^2)
                loss_ctc, min_perm = self.pit.pit_process(loss_ctc_perm)

                # Permute ys
                for b in range(batch_size):  # B
                    ys_pad[:, b] = ys_pad[min_perm[b], b]

            else:
                # (num_spk, B, T, D)
                hs_pad = torch.stack(hs_pad, dim=0)
                for b in range(batch_size):  # B
                    #ys_pad[:, b] = ys_pad[min_perm[b], b]
                    hs_pad[:, b] = hs_pad[min_perm[b], b]
                hs_pad = torch.unbind(hs_pad, dim=0)
                loss_ctc = torch.stack([
                    self.ctc(hs_pad[i], hs_len[i], ys_pad[i])
                    for i in range(self.num_spkrs)
                ], dim=1).mean()

        if float(loss_ctc) >= CTC_LOSS_THRESHOLD:
            logging.warning('Abnormal CTC loss detected: ' + str(float(loss_ctc)))
        else:
            logging.info('ctc loss:' + str(float(loss_ctc)))

        # 2. forward decoder
        if not isinstance(hs_pad, (list, tuple)): # single-speaker input xs_pad
            pred_pad, pred_mask, loss_att, acc, cer_ctc = self.decoder_and_attention(hs_pad, hs_mask, ys_pad, batch_size)
        else:  # multi-speaker input xs_pad
            assert batch_size == ys_pad.size(1)
#            for b in range(batch_size):  # B
#                ys_pad[:, b] = ys_pad[min_perm[b], b]
            pred_pad, pred_mask, loss_att, acc, cer_ctc = [], [], [], [], []
            for i in range(self.num_spkrs):
                p1, p2, l_a, ac, ce = self.decoder_and_attention(hs_pad[i], hs_mask[i], ys_pad[i], batch_size)
                pred_pad.append(p1)
                pred_mask.append(p2)
                loss_att.append(l_a)
                acc.append(ac)
                cer_ctc.append(ce)
            ys_out_len = [float(torch.sum(ys_pad[i] != self.ignore_id)) for i in range(self.num_spkrs)]
            loss_att = sum(map(lambda x: x[0] * x[1], zip(loss_att, ys_out_len))) / sum(ys_out_len)
            acc = sum(map(lambda x: x[0] * x[1], zip(acc, ys_out_len))) / sum(ys_out_len)
            if self.error_calculator is not None:
                cer_ctc = sum(map(lambda x: x[0] * x[1], zip(cer_ctc, ys_out_len))) / sum(ys_out_len)
        self.pred_pad = pred_pad
        self.loss_att = loss_att
        self.acc = acc

        # 5. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            if not isinstance(pred_pad, (list, tuple)):
                ys_hat = pred_pad.argmax(dim=-1)
                cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
            else:
                rslt = []
                for i in range(self.num_spkrs):
                    ys_hat = pred_pad[i].argmax(dim=-1)
                    rslt.append(self.error_calculator(ys_hat.cpu(), ys_pad[i].cpu()))
                cer = sum([r[0] for r in rslt]) / float(len(rslt))
                wer = sum([r[1] for r in rslt]) / float(len(rslt))

        # copyied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        if self.bp_enh_loss and loss_enh is not None:
            self.loss = self.loss + loss_enh * self.enh_loss_weight
            loss_enh_data = float(loss_enh)
        else:
            loss_enh_data = None

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_enh_data, loss_ctc_data, loss_att_data, acc, cer, wer, loss_data)
#            self.reporter.report(loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def decoder_and_attention(self, hs_pad, hs_mask, ys_pad, batch_size):
        """forward decoder and attention loss."""
        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        if self.error_calculator is not None:
            ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        else:
            cer_ctc = None

        # forward decoder
        ys_in_pad, ys_out_pad = self.add_sos_eos(ys_pad)
        ys_mask = self.target_mask(ys_in_pad)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        #compute attention loss
        loss_att = self.criterion(pred_pad, ys_out_pad)
        acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                            ignore_label=self.ignore_id)

        return pred_pad, pred_mask, loss_att, acc, cer_ctc

    def scorers(self):
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, feat):
        self.eval()
        feat = torch.as_tensor(feat).unsqueeze(0)
        enc_output, _ = self.encoder(feat, None)
        return enc_output.squeeze(0)

    def encode_raw(self, x):
        """Encode acoustic features.
        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        h = to_device(self, to_torch_tensor(x).float())
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)
        # 0. Frontend
        if self.frontend is not None:
            hs, hlens, mask = self.frontend(hs, ilens)
            hlens_n = [None] * self.num_spkrs
            for i in range(self.num_spkrs):
                hs[i], hlens_n[i] = self.feature_transform(hs[i].float(), hlens)
            hlens = hlens_n
        else:
            hs, hlens = self.feature_transform(hs.float(), ilens)

        # Encoder
        if isinstance(hs, (list, tuple)):
            hs_pad = [None] * self.num_spkrs
            src_mask = [None] * self.num_spkrs
            for i in range(self.num_spkrs):
                hs_pad[i] = hs[i][:, :max(hlens[i])]  # for data parallel
                src_mask[i] = (~make_pad_mask(hlens[i].tolist())).to(hs_pad[i].device).unsqueeze(-2)
                hs_pad[i], _ = self.encoder(hs_pad[i], src_mask[i])
            enc_output = hs_pad
        else:
            src_mask = make_non_pad_mask(hlens.tolist()).to(hs.device).unsqueeze(-2)
            enc_output, _ = self.encoder(hs, src_mask)
        return enc_output

    def recog(self, enc_output, recog_args, char_list=None, rnnlm=None, use_jit=False):
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info('input lengths: ' + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        
        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, np)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.recognize, (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)
                else:
                    local_att_scores = self.decoder.recognize(ys, ys_mask, enc_output)

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(feat, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps

    def recognize(self, feat, recog_args, char_list=None, rnnlm=None, use_jit=False):
        '''recognize feat

        :param ndnarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list

        TODO(karita): do not recompute previous attention for faster decoding
        '''
        prev = self.training
        self.eval()
        ilens = [feat.shape[0]]

        h = to_device(self, to_torch_tensor(feat).float())
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 0. forward frontend
        if self.frontend is not None:
            hs, hlens, mask = self.frontend(hs, ilens)
            if isinstance(hs, (list, tuple)):
                hlens_n = [None] * self.num_spkrs
                for i in range(self.num_spkrs):
                    hs[i], hlens_n[i] = self.feature_transform(hs[i].float(), hlens)
                hlens = hlens_n
            else:
                hs, hlens = self.feature_transform(hs.float(), hlens)
        else:
            hs, hlens = hs.float(), ilens

        #enc_output = self.encode(hs_pad).unsqueeze(0)
        #enc_output, _ = self.encoder(hs_pad, None)
        # 1. Encoder
        if not isinstance(hs, (list, tuple)):  # single-channel multi-speaker input x
            enc_output, _ = self.encoder(hs, None)

            nbest_hyps = []
            nbest_hyps.append(self.recog(enc_output, recog_args, char_list, rnnlm, use_jit))
            return nbest_hyps
        else:  # multi-channel multi-speaker input x
            enc_output = [None] * self.num_spkrs
            for i in range(self.num_spkrs):
                enc_output[i], _ = self.encoder(hs[i], None)

            nbest_hyps = []
            for enc_out in enc_output:
                nbest_hyps.append(self.recog(enc_out, recog_args, char_list, rnnlm, use_jit))
            return nbest_hyps

    def enhance(self, feat):
        """Forward only the frontend stage.

        :param ndarray feat: input acoustic feature (T, C, F)
        """
        if self.frontend is None:
            raise RuntimeError('Frontend doesn\'t exist')
        prev = self.training
        self.eval()
        ilens = np.fromiter((f.shape[0] for f in feat), dtype=np.int64)

        xs = [to_device(self, to_torch_tensor(f).float()) for f in feat]
        xs_pad = pad_list(xs, 0.0)
        enhanceds, hlens, masks = self.frontend(xs_pad, ilens)
        if prev:
            self.train()

        if isinstance(enhanceds, (tuple, list)):
            enhanceds = list(enhanceds)
            masks = list(masks)
            for idx in range(len(enhanceds)):  # number of speakers
                enhanceds[idx] = enhanceds[idx].cpu().numpy()
                masks[idx] = masks[idx].cpu().numpy()
            return enhanceds, masks, ilens
        else:
            return enhanceds.cpu().numpy(), masks.cpu().numpy(), ilens

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, tgt_pad=None, wav_lens=None):
        '''E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        with torch.no_grad():
            # 0. Frontend
            if self.frontend is not None:
                xs_pad = to_torch_tensor(xs_pad)
                hs_pad, hlens, mask = self.frontend(xs_pad, ilens)
                if isinstance(hs_pad, (list, tuple)):
                    if tgt_pad is not None:
                        tgt_pad = to_device(self, to_torch_tensor(tgt_pad))
                        #with torch.no_grad():
                        #    loss_enh_perm = torch.stack(
                        #        [
                        #            tf_mse_loss(
                        #                hs_pad[i // self.num_spkrs],
                        #                tgt_pad[i % self.num_spkrs]
                        #            )
                        #            for i in range(self.num_spkrs ** 2)
                        #        ],
                        #    dim=1)  # (B, num_spkrs^2)
                        #loss_enh, min_perm = self.pit.pit_process(loss_enh_perm)
                        loss_enh, min_perm = compute_enh_loss(
                            xs_pad, wav_lens, tgt_pad, hs_pad, mask, self.enh_loss_type,
                            mask_type=self.mask_type, stft=self.stft, ref_channel=self.ref_channel,
                        )
                        # scale by input length
                        #loss_enh = loss_enh * hs_pad[0].shape[1]
                    else:
                        loss_enh, min_perm = None, None

                    hlens_n = [None] * self.num_spkrs
                    for i in range(self.num_spkrs):
                        hs_pad[i], hlens_n[i] = self.feature_transform(hs_pad[i].float(), hlens)
                    hlens = hlens_n
                else:
                    hs_pad, hlens = self.feature_transform(hs_pad.float(), hlens)
                    loss_enh, min_perm = None, None
            else:
                hs_pad, hlens = xs_pad.float(), ilens
                loss_enh, min_perm = None, None

            # 1. forward encoder
            if not isinstance(hs_pad, (list, tuple)):  # single-channel input xs_pad (single-speaker)
                num_spkrs = 1
                ret = [dict()] * num_spkrs
                hs_pad = hs_pad[:, :max(hlens)]  # for data parallel
                src_mask = (~make_pad_mask(hlens.tolist())).to(hs_pad.device).unsqueeze(-2)
                hs_pad, hs_mask = self.encoder(hs_pad, src_mask)
                # Encoder attention
                for name, m in self.named_modules():
                    if ('encoder' in name) and isinstance(m, MultiHeadedAttention):
                        ret[0][name] = m.attn.cpu().numpy()
            else:  # multi-channel multi-speaker input xs_pad
                num_spkrs = 2
                ret = [dict()] * num_spkrs
                src_mask = [None] * self.num_spkrs
                hs_mask = [None] * self.num_spkrs
                for i in range(self.num_spkrs):
                    hs_pad[i] = hs_pad[i][:, :max(hlens[i])]  # for data parallel
                    src_mask[i] = (~make_pad_mask(hlens[i].tolist())).to(hs_pad[i].device).unsqueeze(-2)
                    hs_pad[i], hs_mask[i] = self.encoder(hs_pad[i], src_mask[i])
                    # Encoder attention
                    for name, m in self.named_modules():
                        if ('encoder' in name) and isinstance(m, MultiHeadedAttention):
                            ret[i][name] = m.attn.cpu().numpy()
            #self.hs_pad = hs_pad

            assert self.mtlalpha > 0.0 and self.mtlalpha < 1.0
            batch_size = len(ilens)  # hs_pad.size(0)
            if num_spkrs == 1:
                # CTC loss
                hs_len = hs_mask.view(batch_size, -1).sum(1)
                loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)

                # 2. forward decoder
                p1, p2, l_a, ac, ce = self.decoder_and_attention(hs_pad, hs_mask, ys_pad, batch_size)
                # Decoder attention
                for name, m in self.named_modules():
                    if ('decoder' in name) and isinstance(m, MultiHeadedAttention):
                        ret[0][name] = m.attn.cpu().numpy()
            elif num_spkrs == 2:
                # CTC loss
                ys_pad = ys_pad.transpose(0, 1)  # (num_spkrs, B, Lmax)
                hs_len = [hs_mask[i].view(batch_size, -1).sum(1) for i in range(self.num_spkrs)]
                if min_perm is None:
                    loss_ctc_perm = torch.stack([self.ctc(hs_pad[i // self.num_spkrs].view(batch_size, -1, self.adim),
                                                        hs_len[i // self.num_spkrs],
                                                        ys_pad[i % self.num_spkrs])
                                                for i in range(self.num_spkrs ** 2)], dim=1)  # (B, num_spkrs^2)
                    loss_ctc, min_perm = self.pit.pit_process(loss_ctc_perm)

                    # Permute ys
                    for b in range(batch_size):  # B
                        ys_pad[:, b] = ys_pad[min_perm[b], b]

                else:
                    for b in range(batch_size):  # B
                        ys_pad[:, b] = ys_pad[min_perm[b], b]
                    loss_ctc = torch.stack([
                        self.ctc(hs_pad[i], hs_len[i], ys_pad[i])
                        for i in range(self.num_spkrs)
                    ], dim=1).mean()

                # 2. forward decoder
                for i in range(self.num_spkrs):
                    p1, p2, l_a, ac, ce = self.decoder_and_attention(hs_pad[i], hs_mask[i], ys_pad[i], batch_size)
                    # Decoder attention
                    for name, m in self.named_modules():
                        if ('decoder' in name) and isinstance(m, MultiHeadedAttention):
                            ret[i][name] = m.attn.cpu().numpy()
            else:
                raise ValueError('Unexpected number of speakers: {}'.format(num_spkrs))
        return ret
