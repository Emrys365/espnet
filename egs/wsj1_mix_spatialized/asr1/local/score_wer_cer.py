#!/usr/bin/env python3
# Calculate WER and CER from data.json derived from exclusive decoding process


import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode-json', type=str,
                        help='path to data.json in the decoding directory')
    args = parser.parse_args()

    wers, cers = [], []
    with open(args.decode_json, 'r') as f:
        results = json.load(f)['utts']

    wers, cers = [], []
    dists_word, dists_char = [], []
    ns_word, ns_char = [], []
    for uttid, dic in results.items():
        wers.append(dic['output'][0]['wer'])
        cers.append(dic['output'][0]['cer'])
        dists_word.append(dic['output'][0]['ed_word'])
        dists_char.append(dic['output'][0]['ed_char'])
        ns_word.append(dic['output'][0]['n_word'])
        ns_char.append(dic['output'][0]['n_char'])

    mean_wer = float(sum(wers)) / len(wers)
    mean_cer = float(sum(cers)) / len(cers)
    WER = float(sum(dists_word)) / sum(ns_word)
    CER = float(sum(dists_char)) / sum(ns_char)
    print('mean WER: {}\nmean CER: {}'.format(mean_wer, mean_cer))
    print('total WER: {}\ntotal CER: {}'.format(WER, CER))
