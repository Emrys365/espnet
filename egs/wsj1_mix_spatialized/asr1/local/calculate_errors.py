#!/usr/bin/env python3
# Calculate WER and CER from data.json derived from normal decoding process

import argparse
import editdistance
import json
import torch

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


def compute_word_char_edit_dist(rec_texts, true_texts):
    """Compute edit distance in the word level and character level

    Args:
        rec_texts: List of recognized text
        true_texts: List of target text
    Returns:
        ed_word: sum of pairwise word-level edit distance
        ed_char: sum of pairwise char-level edit distance
        num_words: total number of words in `true_texts`
        num_chars: total number of characters in `true_texts`
        wer: sequence-level WER
        cer: sequence-level CER
    """
    assert len(rec_texts) == len(true_texts)
    num_chars = len(true_texts)

    hyp_words, hyp_chars = [], []
    ref_words, ref_chars = [], []
    for ns in range(num_spkrs):
        hyp_words.append(rec_texts[ns].split())
        ref_words.append(true_texts[ns].split())
        hyp_chars.append(rec_texts[ns].replace(' ', ''))
        ref_chars.append(true_texts[ns].replace(' ', ''))

    tmp_word_ed = [editdistance.eval(hyp_words[ns // num_spkrs], ref_words[ns % num_spkrs])
                   for ns in range(num_spkrs ** 2)]  # h1r1,h1r2,h2r1,h2r2
    tmp_char_ed = [editdistance.eval(hyp_chars[ns // num_spkrs], ref_chars[ns % num_spkrs])
                   for ns in range(num_spkrs ** 2)]  # h1r1,h1r2,h2r1,h2r2

    pit = PIT(num_spkrs)
    min_word_ed, perm_word = pit.min_pit_sample(torch.tensor(tmp_word_ed))
    min_char_ed, perm_char = pit.min_pit_sample(torch.tensor(tmp_char_ed))
    num_words = len(sum(ref_words, []))
    num_chars = len(''.join(ref_chars))
    wer = float(min_word_ed) / num_words
    cer = float(min_char_ed) / num_chars
    return int(min_word_ed), int(min_char_ed), num_words, num_chars, wer, cer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode-json', type=str,
                        help='path to data.json in the decoding directory')
    args = parser.parse_args()

    wers, cers = [], []
    with open(args.decode_json, 'r') as f:
        results = json.load(f)['utts']

    num_spkrs = 2
    wers, cers = [], []
    dists_word, dists_char = [], []
    ns_word, ns_char = [], []
    for uttid, dic in results.items():
        true_texts = []
        rec_texts = []
        if 'output' in dic:
            for n in range(1, num_spkrs + 1):
                true_texts.append(dic['output'][n]['text'])
                rec_texts.append(dic['output'][n]['rec_text'])
        elif 'output1' in dic:
            for n in range(1, num_spkrs + 1):
                true_texts.append(dic['output%d' % n][0]['text'])
                rec_texts.append(dic['output%d' % n][0]['rec_text'])
        else:
            raise KeyError('no matched key for "output" or "output1": ' + str(dic.keys()))

        ed_word, ed_char, num_words, num_chars, wer, cer = \
            compute_word_char_edit_dist(rec_texts, true_texts)

        dic['results'] = {
            'wer': wer, 'cer': cer,
            'ed_word': ed_word,
            'ed_char': ed_char,
            'ns_word': num_words,
            'ns_char': num_chars
        }

        dists_word.append(ed_word)
        dists_char.append(ed_char)
        ns_word.append(num_words)
        ns_char.append(num_chars)
        wers.append(wer)
        cers.append(cer)

    mean_wer = float(sum(wers)) / len(wers)
    mean_cer = float(sum(cers)) / len(cers)
    WER = float(sum(dists_word)) / sum(ns_word)
    CER = float(sum(dists_char)) / sum(ns_char)
    print('mean WER: {}\nmean CER: {}'.format(mean_wer, mean_cer))
    print('total word-level edit distance: {}\ntotal char-level edit distance: {}'.format(
          sum(dists_word), sum(dists_char)))
    print('total words: {}\ntotal characters: {}'.format(sum(ns_word), sum(ns_char)))
    print('total WER: {}\ntotal CER: {}'.format(WER, CER))
    with open('tmp.json', 'wb') as f:
        f.write(json.dumps({'utts': results}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
