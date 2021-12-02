#!/usr/bin/env python3
import argparse
import json
import re

from dict_spk2gender import WSJ0_spk2gender

# Usage:
#  ./get_breakdown.py [RESULT_JSON] [DATA_JSON] --skip 4 [--utt2rt60 []]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_json', type=str,
                        help='path to min_perm_result.*.json in the decoding dir')
    parser.add_argument('data_json', type=str,
                        help='path to data.json in the dump dir')
    parser.add_argument('--skip', type=int, default=0,
                        help='skip first N rows of `results_json`')
    parser.add_argument('--utt2rt60', type=str, default=None,
                        help='utt2rt60 file containing the RT60 for each utterance')
    args = parser.parse_args()

    with open(args.data_json, 'r') as f:
        data = json.load(f)['utts']

    if args.utt2rt60 is not None:
        utt2rt60 = {}
        with open(args.utt2rt60, 'r') as f:
            # 440c0404_-2.0147_441c040i_2.0147.wav
            for line in f:
                if len(line.strip()) > 0:
                    uttid, rt60 = line.strip().split(maxsplit=1)
                    utt2rt60[uttid] = float(rt60)
    else:
        utt2rt60 = None

    with open(args.result_json, 'r') as f:
        lines = f.readlines()

    results = json.loads('\n'.join(lines[args.skip:]))['utts']
    keys = results.keys()

    new_dic = {}
    for gt in ['MM', 'FF', 'MF']:
        new_dic[gt] = {}
        for st in ['LowSNR', 'HighSNR']:
            new_dic[gt][st] = {}
            for spk in ['HighE', 'LowE']:
                new_dic[gt][st][spk] = []

    length_dic = {}
    for lt in ['Long', 'Short']:
        length_dic[lt] = {}
        for st in ['LowSNR', 'HighSNR']:
            length_dic[lt][st] = {}
            for spk in ['HighE', 'LowE']:
                length_dic[lt][st][spk] = []

    input_lengths = [data[k[1:-1].split('-', 1)[1]]['input'][0]['shape'][0] for k in keys]
#    mean_length = sum(input_lengths) / float(len(input_lengths))
    mean_length = 998.5
    min_length, max_length = min(input_lengths), max(input_lengths)
    if utt2rt60 is not None:
        rt60s = utt2rt60.values()
        mean_rt60 = sum(rt60s) / len(rt60s)
        min_rt60, max_rt60 = min(rt60s), max(rt60s)
        print('mean RT60: {}\nmin RT60: {}\nmax RT60: {}'.format(mean_rt60, min_rt60, max_rt60))
        rt60_dic = {}
        for rt in ['HighReverb', 'LowReverb']:
            rt60_dic[rt] = {}
            for st in ['LowSNR', 'HighSNR']:
                rt60_dic[rt][st] = {}
                for spk in ['HighE', 'LowE']:
                    rt60_dic[rt][st][spk] = []

    for k in keys:
        # key: (xxx_yyy-xxx_yyy_xxxabcd_snr_yyyefgh_-snr)
        spkr_ids, utt_id = k[1:-1].split('-', 1)
        length = data[utt_id]['input'][0]['shape'][0]
        len_type = 'Long' if length > mean_length else 'Short'

        if utt2rt60 is not None:
            rt60_type = 'HighReverb' if utt2rt60[utt_id] > mean_rt60 else 'LowReverb'

        lst = utt_id.split('_')
        snr1, snr2 = float(lst[3]), float(lst[5])
        snr = abs(snr1 - snr2)  # 0 ~ 5 dB
        if snr <= 2.5:
            snr_type = 'LowSNR'
        else:
            snr_type = 'HighSNR'

        spkr1_id, spkr2_id = spkr_ids.split('_', 1)
        gender1, gender2 = WSJ0_spk2gender[spkr1_id], WSJ0_spk2gender[spkr2_id]
        if gender1 == gender2:
            gender_type = gender1 + gender2
        else:
            gender_type = "MF"

        num_correct, num_sub, num_del, num_ins = \
            map(int, re.split(r'\s+', results[k]['Scores'].split(')', 1)[1].strip()))

        if 'r1h2' in results[k]:
            k1, k2 = 'r1h2', 'r2h1'
        elif 'r1h1' in results[k]:
            k1, k2 = 'r1h1', 'r2h2'
        else:
            raise KeyError('Invalid keys: {}'.format(results[k].keys()))

        num_correct1, num_sub1, num_del1, num_ins1 = \
            map(int, re.split(r'\s+', results[k][k1]['Scores'].split(')', 1)[1].strip()))
        num_correct2, num_sub2, num_del2, num_ins2 = \
            map(int, re.split(r'\s+', results[k][k2]['Scores'].split(')', 1)[1].strip()))

        err = float(num_sub + num_del + num_ins) / (num_correct + num_sub + num_del + num_ins)
        err1 = float(num_sub1 + num_del1 + num_ins1) / (num_correct1 + num_sub1 + num_del1 + num_ins1)
        err2 = float(num_sub2 + num_del2 + num_ins2) / (num_correct2 + num_sub2 + num_del2 + num_ins2)

        if snr1 >= snr2:
            new_dic[gender_type][snr_type]['HighE'].append(err1)
            new_dic[gender_type][snr_type]['LowE'].append(err2)
            length_dic[len_type][snr_type]['HighE'].append(err1)
            length_dic[len_type][snr_type]['LowE'].append(err2)
            if utt2rt60 is not None:
                rt60_dic[rt60_type][snr_type]['HighE'].append(err1)
                rt60_dic[rt60_type][snr_type]['LowE'].append(err2)
        else:
            new_dic[gender_type][snr_type]['HighE'].append(err2)
            new_dic[gender_type][snr_type]['LowE'].append(err1)
            length_dic[len_type][snr_type]['HighE'].append(err2)
            length_dic[len_type][snr_type]['LowE'].append(err1)
            if utt2rt60 is not None:
                rt60_dic[rt60_type][snr_type]['HighE'].append(err2)
                rt60_dic[rt60_type][snr_type]['LowE'].append(err1)


    for gt in ['FF', 'MM', 'MF']:
        for st in ['LowSNR', 'HighSNR']:
            ret = "{} {}\t".format(gt, st)
            for spk in ['HighE', 'LowE']:
                mean_err = sum(new_dic[gt][st][spk]) / len(new_dic[gt][st][spk])
                ret += '{}: {:2.2f} % ({} samples)\t'.format(spk, mean_err * 100, len(new_dic[gt][st][spk]))
            print(ret)

    print('')
    for rt in ['LowReverb', 'HighReverb']:
        for st in ['LowSNR', 'HighSNR']:
            ret = "{} {}\t".format(rt, st)
            for spk in ['HighE', 'LowE']:
                mean_err = sum(rt60_dic[rt][st][spk]) / len(rt60_dic[rt][st][spk])
                ret += '{}: {:2.2f} % ({} samples)\t'.format(spk, mean_err * 100, len(rt60_dic[rt][st][spk]))
            print(ret)

    win_length = 0.025 * 16000
    win_shift = 0.01 * 16000
    mean_time = (mean_length * win_shift + win_length - win_shift) / 16000
    min_time = (min_length * win_shift + win_length - win_shift) / 16000
    max_time = (max_length * win_shift + win_length - win_shift) / 16000
    print('\nmean length: {} ≈ {}s'.format(mean_length, mean_time))
    print('min length: {} ≈ {}s'.format(min_length, min_time))
    print('max length: {} ≈ {}s'.format(max_length, max_time))
    for lt in ['Short', 'Long']:
        for st in ['LowSNR', 'HighSNR']:
            ret = "{} {}\t".format(lt, st)
            for spk in ['HighE', 'LowE']:
                mean_err = sum(length_dic[lt][st][spk]) / len(length_dic[lt][st][spk])
                ret += '{}: {:2.2f}% ({} samples)\t'.format(spk, mean_err * 100, len(length_dic[lt][st][spk]))
            print(ret)
