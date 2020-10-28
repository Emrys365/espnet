"""
This script is used to construct End-to-End models of multi-speaker ASR with transformers.

Copyright 2019 Shigeki Karita
          2019 Xuankai Chang
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from argparse import Namespace
from distutils.util import strtobool

import logging
import math
import chainer
import numpy as np

import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr_mix import PIT
from espnet.nets.pytorch_backend.e2e_asr_mix import Reporter
from espnet.nets.pytorch_backend.e2e_asr_mix import E2E as E2E_ASR_Mix
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2E_ASR
from espnet.nets.pytorch_backend.frontends.feature_transform import feature_transform_for
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
        group.add_argument('--wpd-opt', type=float, default=1, choices=[1, 2, 3, 4, 5, 5.2, 5.3, 6],
                           help='which WPD implementation to be used')
        group.add_argument('--use-padertorch-frontend', type=strtobool, default=False,
                           help='use padertorch-like frontend')
        group.add_argument('--use-vad-mask', type=strtobool, default=False,
                           help='use VAD-like masks instead of T-F masks, only works when use_padertorch_frontend is True')
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
                from espnet.nets.pytorch_backend.frontends.frontend_v2 import frontend_for
                logging.warning('Using padertorch-like frontend')
            else:
                from espnet.nets.pytorch_backend.frontends.frontend import frontend_for

            self.frontend = frontend_for(args, idim)
            self.feature_transform = feature_transform_for(args, (idim - 1) * 2)
            idim = args.n_mels + 3 if getattr(args, "fbank_pitch", None) is not None else args.n_mels
        else:
            self.frontend = None

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
            self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=False)
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

    def forward(self, xs_pad, ilens, ys_pad):
        '''E2E forward

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
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
            hs_pad, hlens, mask = self.frontend(xs_pad, ilens)
            if isinstance(hs_pad, list):
                hlens_n = [None] * self.num_spkrs
                for i in range(self.num_spkrs):
                    hs_pad[i], hlens_n[i] = self.feature_transform(hs_pad[i].float(), hlens)
                hlens = hlens_n
            else:
                # move to GPU for ASR
                hs_pad, hlens = self.feature_transform(hs_pad.float(), hlens)
        else:
            hs_pad, hlens = xs_pad.float(), ilens

        # 1. forward encoder
        if not isinstance(hs_pad, list):  # single-channel input xs_pad (single-speaker)
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
        if not isinstance(hs_pad, list):  # single-speaker input xs_pad
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = torch.mean(self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad))
        else:  # multi-speaker input xs_pad
            ys_pad = ys_pad.transpose(0, 1)  # (num_spkrs, B, Lmax)
            hs_len = [hs_mask[i].view(batch_size, -1).sum(1) for i in range(self.num_spkrs)]
            loss_ctc_perm = torch.stack([self.ctc(hs_pad[i // self.num_spkrs].view(batch_size, -1, self.adim),
                                                  hs_len[i // self.num_spkrs],
                                                  ys_pad[i % self.num_spkrs])
                                         for i in range(self.num_spkrs ** 2)], dim=1)  # (B, num_spkrs^2)
            loss_ctc, min_perm = self.pit.pit_process(loss_ctc_perm)

        if float(loss_ctc) >= CTC_LOSS_THRESHOLD:
            logging.warning('Abnormal CTC loss detected: ' + str(float(loss_ctc)))
        else:
            logging.info('ctc loss:' + str(float(loss_ctc)))

        # 2. forward decoder
        if not isinstance(hs_pad, list): # single-speaker input xs_pad
            pred_pad, pred_mask, loss_att, acc, cer_ctc = self.decoder_and_attention(hs_pad, hs_mask, ys_pad, batch_size)
        else:  # multi-speaker input xs_pad
            assert batch_size == ys_pad.size(1)
            for b in range(batch_size):  # B
                ys_pad[:, b] = ys_pad[min_perm[b], b]
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
            if not isinstance(pred_pad, list):
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

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, loss_att_data, acc, cer, wer, loss_data)
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
        if isinstance(hs, list):
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
            ctc_prefix_score = CTCPrefixScore(lpz.detach().cpu().numpy(), 0, self.eos, np)
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
                    local_att_scores = self.decoder.recognize(
                        ys.to(enc_output.device), ys_mask.to(enc_output.device), enc_output.to(enc_output.device)
                    )

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
            if isinstance(hs, list):
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
        if not isinstance(hs, list):  # single-channel multi-speaker input x
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

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
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
                if isinstance(hs_pad, list):
                    hlens_n = [None] * self.num_spkrs
                    for i in range(self.num_spkrs):
                        hs_pad[i], hlens_n[i] = self.feature_transform(hs_pad[i].float(), hlens)
                    hlens = hlens_n
                else:
                    hs_pad, hlens = self.feature_transform(hs_pad.float(), hlens)
            else:
                hs_pad, hlens = xs_pad.float(), ilens

            # 1. forward encoder
            if not isinstance(hs_pad, list):  # single-channel input xs_pad (single-speaker)
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
                loss_ctc_perm = torch.stack([self.ctc(hs_pad[i // self.num_spkrs].view(batch_size, -1, self.adim),
                                                    hs_len[i // self.num_spkrs],
                                                    ys_pad[i % self.num_spkrs])
                                            for i in range(self.num_spkrs ** 2)], dim=1)  # (B, num_spkrs^2)
                loss_ctc, min_perm = self.pit.pit_process(loss_ctc_perm)
                # Permute ys
                for b in range(batch_size):  # B
                    ys_pad[:, b] = ys_pad[min_perm[b], b]

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
