#!/usr/bin/env python3

"""
This script is used to provide utility functions designed for multi-speaker ASR.

Copyright 2017 Johns Hopkins University (Shinji Watanabe)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

Most functions can be directly used as in asr_utils.py:
    CompareValueTrigger, restore_snapshot, adadelta_eps_decay, chainer_load,
    torch_snapshot, torch_save, torch_resume, AttributeDict, get_model_conf.

"""

import copy
import editdistance
import logging
import os

from chainer.training import extension

import matplotlib

from espnet.asr.asr_utils import parse_hypothesis
#from espnet.nets.pytorch_backend.e2e_asr_mix import PIT


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
                                       loss[1] + loss[2]])
        elif self.num_spkrs == 3:
            score_perms = torch.stack([loss[0] + loss[4] + loss[8],
                                       loss[0] + loss[5] + loss[7],
                                       loss[1] + loss[5] + loss[6],
                                       loss[1] + loss[3] + loss[8],
                                       loss[2] + loss[3] + loss[7],
                                       loss[2] + loss[4] + loss[6]])

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
        bs = losses.size(0)
        ret = [self.min_pit_sample(losses[i]) for i in range(bs)]

        loss_perm = torch.stack([r[0] for r in ret], dim=0).to(losses.device)  # (B)
        permutation = torch.tensor([r[1] for r in ret]).long().to(losses.device)

        return torch.mean(loss_perm), permutation


matplotlib.use('Agg')


# * -------------------- chainer extension related -------------------- *
class PlotAttentionReport(extension.Extension):
    """Plot attention reporter.

    Args:
        att_vis_fn (espnet.nets.*_backend.e2e_asr.calculate_all_attentions): Function of attention visualization.
        data (list[tuple(str, dict[str, dict[str, Any]])]): List json utt key items.
        outdir (str): Directory to save figures.
        converter (espnet.asr.*_backend.asr.CustomConverter): CustomConverter object. Function to convert data.
        device (torch.device): The destination device to send tensor.
        reverse (bool): If True, input and output length are reversed.

    """

    def __init__(self, att_vis_fn, data, outdir, converter, device, reverse=False):
        """Initialize PlotAttentionReport."""
        self.att_vis_fn = att_vis_fn
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.device = device
        self.reverse = reverse
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        """Plot and save imaged matrix of att_ws."""
        att_ws_sd = self.get_attention_weights()
        for ns, att_ws in enumerate(att_ws_sd):
            for idx, att_w in enumerate(att_ws):
                filename = "%s/%s.ep.{.updater.epoch}.output%d.png" % (
                    self.outdir, self.data[idx][0], ns + 1)
                att_w = self.get_attention_weight(idx, att_w, ns)
                self._plot_and_save_attention(att_w, filename.format(trainer))

    def log_attentions(self, logger, step):
        """Add image files of attention matrix to tensorboard."""
        att_ws_sd = self.get_attention_weights()
        for ns, att_ws in enumerate(att_ws_sd):
            for idx, att_w in enumerate(att_ws):
                att_w = self.get_attention_weight(idx, att_w, ns)
                plot = self.draw_attention_plot(att_w)
                logger.add_figure("%s" % (self.data[idx][0]), plot.gcf(), step)
                plot.clf()

    def get_attention_weights(self):
        """Return attention weights.

        Returns:
            arr_ws_sd (numpy.ndarray): attention weights. It's shape would be
                differ from bachend.dtype=float
                * pytorch-> 1) multi-head case => (B, H, Lmax, Tmax). 2) other case => (B, Lmax, Tmax).
                * chainer-> attention weights (B, Lmax, Tmax).

        """
        batch = self.converter([self.converter.transform(self.data)], self.device)
        att_ws_sd = self.att_vis_fn(*batch)
        return att_ws_sd

    def get_attention_weight(self, idx, att_w, spkr_idx):
        """Transform attention weight in regard to self.reverse."""
        if self.reverse:
            dec_len = int(self.data[idx][1]['input'][0]['shape'][0])
            enc_len = int(self.data[idx][1]['output'][spkr_idx]['shape'][0])
        else:
            dec_len = int(self.data[idx][1]['output'][spkr_idx]['shape'][0])
            enc_len = int(self.data[idx][1]['input'][0]['shape'][0])
        if len(att_w.shape) == 3:
            att_w = att_w[:, :dec_len, :enc_len]
        else:
            att_w = att_w[:dec_len, :enc_len]
        return att_w

    def draw_attention_plot(self, att_w):
        """Visualize attention weights matrix.

        Args:
            att_w(Tensor): Attention weight matrix.

        Returns:
            matplotlib.pyplot: pyplot object with attention matrix image.

        """
        import matplotlib.pyplot as plt
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                plt.subplot(1, len(att_w), h)
                plt.imshow(aw, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
        else:
            plt.imshow(att_w, aspect="auto")
            plt.xlabel("Encoder Index")
            plt.ylabel("Decoder Index")
        plt.tight_layout()
        return plt

    def _plot_and_save_attention(self, att_w, filename):
        plt = self.draw_attention_plot(att_w)
        plt.savefig(filename)
        plt.close()


def add_results_to_json_wer(js, nbest_hyps_sd, ys, char_list):
    """Add N-best results (WER & CER) to json.

    Args:
        js (dict[str, Any]): Groundtruth utterance dict.
        nbest_hyps_sd (list[dict[str, Any]]): List of hypothesis for multi_speakers (# Utts x # Spkrs).
        char_list (list[str]): List of characters.

    Returns:
        dict[str, Any]: N-best results added utterance dict.

    """
    import numpy as np
    import torch
    from espnet.nets.pytorch_backend.nets_utils import pad_list

    if not isinstance(ys[0], np.ndarray):
        ys_pad = [torch.from_numpy(y[0]).long() for y in ys] + [torch.from_numpy(y[1]).long() for y in ys]
        ys_pad = pad_list(ys_pad, -1)
        ys_pad = ys_pad.view(2, -1, ys_pad.size(1))  # (num_spkrs, B, Tmax)
    else:
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], -1).transpose(0, 1)

    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    num_spkrs = len(nbest_hyps_sd)
    new_js['output'] = []
    pit = PIT(num_spkrs)

    # remove <sos> and <eos>
    y_hats = [[nbest_hyp['yseq'][1:] for nbest_hyp in nbest_hyps_sd[i]] for i in range(num_spkrs)]
    for i in range(len(y_hats[0])):
        # iterate through all hypotheses
        # copy ground-truth
        out_dics = [dict(js['output'][ns].items()) for ns in range(num_spkrs)]

        word_eds, char_eds, word_ref_lens, char_ref_lens = [], [], [], []
        hyp_words = []
        hyp_chars = []
        ref_words = []
        ref_chars = []
        rec_tokens, rec_tokenids, rec_texts = [], [], []
        for ns in range(num_spkrs):
            y_hat = y_hats[ns][i]
            y_true = ys_pad[ns][i]

            rec_tokenid_list = [int(idx) for idx in y_hat if int(idx) != -1]
            rec_tokenid = " ".join([str(idx) for idx in rec_tokenid_list])
            rec_tokenids.append(rec_tokenid)

            rec_token_list = [char_list[int(idx)] for idx in y_hat if int(idx) != -1]
            rec_token = " ".join(rec_token_list)
            rec_tokens.append(rec_token)

            rec_text = "".join(rec_token_list).replace('<space>', ' ')
            rec_text = rec_text.replace('<blank>', '').replace('<eos>', '')
            rec_texts.append(rec_text)

            true_token_list = [char_list[int(idx)] for idx in y_true if int(idx) != -1]
            true_text = "".join(true_token_list).replace('<space>', ' ')

            hyp_words.append(rec_text.split())
            ref_words.append(true_text.split())
            hyp_chars.append(rec_text.replace(' ', ''))
            ref_chars.append(true_text.replace(' ', ''))

        tmp_word_ed = [editdistance.eval(hyp_words[ns // num_spkrs], ref_words[ns % num_spkrs])
                       for ns in range(num_spkrs ** 2)]  # h1r1,h1r2,h2r1,h2r2
        tmp_char_ed = [editdistance.eval(hyp_chars[ns // num_spkrs], ref_chars[ns % num_spkrs])
                       for ns in range(num_spkrs ** 2)]  # h1r1,h1r2,h2r1,h2r2

        min_word_eds, perm_word = pit.min_pit_sample(torch.tensor(tmp_word_ed))
        word_eds.append(min_word_eds)
        word_ref_lens.append(len(sum(ref_words, [])))

        min_char_eds, perm_char = pit.min_pit_sample(torch.tensor(tmp_char_ed))
        char_eds.append(min_char_eds)
        char_ref_lens.append(len(''.join(ref_chars)))

        wer = float(sum(word_eds)) / sum(word_ref_lens)
        cer = float(sum(char_eds)) / sum(char_ref_lens)
        logging.info('wer: {}, cer: {}'.format(wer, cer))
        new_js['output'].append({
            'wer': wer, 'cer': cer,
            'perm_word': str(perm_word),
            'perm_char': str(perm_char),
            'ed_word': float(min_word_eds),
            'ed_char': float(min_char_eds),
            'n_word': word_ref_lens[-1],
            'n_char': char_ref_lens[-1]
        })

        for j, p in enumerate(perm_word):
            # update name
            out_dics[j]['name'] += '[%d]' % i
            out_dics[j]['rec_text'] = rec_texts[p]
            out_dics[j]['rec_token'] = rec_tokens[p]
            out_dics[j]['rec_tokenid'] = rec_tokenids[p]
            new_js['output'].append(out_dics[j])

            # show 1-best result
            if i == 0:
                logging.info('groundtruth: %s' % out_dics[j]['text'])
                logging.info('prediction : %s' % out_dics[j]['rec_text'])
        # only add the top hypothesis
        break

    return new_js


def add_results_to_json(js, nbest_hyps_sd, char_list):
    """Add N-best results to json.

    Args:
        js (dict[str, Any]): Groundtruth utterance dict.
        nbest_hyps_sd (list[dict[str, Any]]): List of hypothesis for multi_speakers (# Utts x # Spkrs).
        char_list (list[str]): List of characters.

    Returns:
        dict[str, Any]: N-best results added utterance dict.

    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    num_spkrs = len(nbest_hyps_sd)
    new_js['output'] = []

    for ns in range(num_spkrs):
        tmp_js = []
        nbest_hyps = nbest_hyps_sd[ns]

        for n, hyp in enumerate(nbest_hyps, 1):
            # parse hypothesis
            rec_text, rec_token, rec_tokenid, score = parse_hypothesis(hyp, char_list)

            # copy ground-truth
            out_dic = dict(js['output'][ns].items())

            # update name
            out_dic['name'] += '[%d]' % n

            # add recognition results
            out_dic['rec_text'] = rec_text
            out_dic['rec_token'] = rec_token
            out_dic['rec_tokenid'] = rec_tokenid
            out_dic['score'] = score

            # add to list of N-best result dicts
            tmp_js.append(out_dic)

            # show 1-best result
            if n == 1:
                logging.info('groundtruth: %s' % out_dic['text'])
                logging.info('prediction : %s' % out_dic['rec_text'])

        new_js['output'].append(tmp_js)
    return new_js
