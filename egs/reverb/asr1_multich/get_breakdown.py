#!/usr/bin/env python3
import argparse
import json
import re

from dict_spk2gender import REVERB_spk2gender

# Usage:
#  ./get_breakdown.py [RESULT_TXT] [DATA_JSON] --skip 4 [--utt2rt60 []]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_txt', type=str,
                        help='path to result*.txt in the decoding dir')
    parser.add_argument('data_json', type=str,
                        help='path to data.json in the dump dir')
    parser.add_argument('--skip', type=int, default=0,
                        help='skip first N rows of `results_json`')
    parser.add_argument('--utt2rt60', type=str, default=None,
                        help='utt2rt60 file containing the RT60 for each utterance')
    args = parser.parse_args()

    data = {}
    with open(args.data_json, 'r') as f:
        for k, v in json.load(f)['utts'].items():
            data[k.lower()] = v

    if args.utt2rt60 is not None:
        utt2rt60 = {}
        with open(args.utt2rt60, 'r') as f:
            # t10_RealData_dt_for_8ch_far_room1_t10c020e
            # c49_SimData_dt_for_8ch_near_room3_c49c0210
            for line in f:
                if len(line.strip()) > 0:
                    uttid, rt60 = line.strip().split(maxsplit=1)
                    utt2rt60[uttid] = float(rt60)
    else:
        utt2rt60 = None

    with open(args.result_txt, 'r') as f:
        lines = f.readlines()

    results = {}
    new_utt = False
    uid = ""
    for line in lines:
        if new_utt:
            scores = map(
                int,
                re.match(
                    r'Scores:\s*\(#C\s+#S\s+#D\s+#I\)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
                    line.strip(),
                ).groups(),
            )
            results[uid] = scores
            new_utt = False
            uid = ""
        else:
            new_utt = re.match(r'id:\s*\(\s*(.*)\s*\)', line.strip())
            if new_utt:
                uid = new_utt.group(1)

    keys = results.keys()

    new_dic = {}
    for gt in ['M', 'F']:
        new_dic[gt] = {}
        for room in [
            'simdata_dt_far_room1', 'simdata_dt_far_room2', 'simdata_dt_far_room3',
            'simdata_dt_near_room1', 'simdata_dt_near_room2', 'simdata_dt_near_room3',
            'simdata_et_far_room1', 'simdata_et_far_room2', 'simdata_et_far_room3',
            'simdata_et_near_room1', 'simdata_et_near_room2', 'simdata_et_near_room3',
            'realdata_dt_far_room1', 'realdata_dt_near_room1',
            'realdata_et_far_room1', 'realdata_et_near_room1',
        ]:
            new_dic[gt][room] = []

    length_dic = {}
    for lt in ['Long', 'Short']:
        length_dic[lt] = {}
        for room in [
            'simdata_dt_far_room1', 'simdata_dt_far_room2', 'simdata_dt_far_room3',
            'simdata_dt_near_room1', 'simdata_dt_near_room2', 'simdata_dt_near_room3',
            'simdata_et_far_room1', 'simdata_et_far_room2', 'simdata_et_far_room3',
            'simdata_et_near_room1', 'simdata_et_near_room2', 'simdata_et_near_room3',
            'realdata_dt_far_room1', 'realdata_dt_near_room1',
            'realdata_et_far_room1', 'realdata_et_near_room1',
        ]:
            length_dic[lt][room] = []

    room_dic = {
        'simdata_dt_far_room1': [],
        'simdata_dt_far_room2': [],
        'simdata_dt_far_room3': [],
        'simdata_dt_near_room1': [],
        'simdata_dt_near_room2': [],
        'simdata_dt_near_room3': [],
        'simdata_et_far_room1': [],
        'simdata_et_far_room2': [],
        'simdata_et_far_room3': [],
        'simdata_et_near_room1': [],
        'simdata_et_near_room2': [],
        'simdata_et_near_room3': [],
        'realdata_dt_far_room1': [],
        'realdata_dt_near_room1': [],
        'realdata_et_far_room1': [],
        'realdata_et_near_room1': [],
    }

    input_lengths = [data[k.split('-', 1)[1]]['input'][0]['shape'][0] for k in keys]
    # mean_length = sum(input_lengths) / float(len(input_lengths))
    mean_length = 672.7
    min_length, max_length = min(input_lengths), max(input_lengths)
    if utt2rt60 is not None:
        rt60s = utt2rt60.values()
        mean_rt60 = sum(rt60s) / len(rt60s)
        min_rt60, max_rt60 = min(rt60s), max(rt60s)
        print('mean RT60: {}\nmin RT60: {}\nmax RT60: {}'.format(mean_rt60, min_rt60, max_rt60))
        rt60_dic = {}
        for rt in ['HighReverb', 'LowReverb']:
            rt60_dic[rt] = {}
            for room in [
                'simdata_dt_far_room1', 'simdata_dt_far_room2', 'simdata_dt_far_room3',
                'simdata_dt_near_room1', 'simdata_dt_near_room2', 'simdata_dt_near_room3',
                'simdata_et_far_room1', 'simdata_et_far_room2', 'simdata_et_far_room3',
                'simdata_et_near_room1', 'simdata_et_near_room2', 'simdata_et_near_room3',
                'realdata_dt_far_room1', 'realdata_dt_near_room1',
                'realdata_et_far_room1', 'realdata_et_near_room1',
            ]:
                rt60_dic[rt][room] = {}

    for k in keys:
        # key: <spk>_<Real_or_Simu_Data>_<subset>_for_8ch_<far_or_near_room?>_<uid>
        spkr_ids, utt_id = k.split('-', 1)
        length = data[utt_id]['input'][0]['shape'][0]
        len_type = 'Long' if length > mean_length else 'Short'

        if utt2rt60 is not None:
            rt60_type = 'HighReverb' if utt2rt60[utt_id] > mean_rt60 else 'LowReverb'

        lst = utt_id.split('_')
        datatype = re.match(r'(RealData|SimData)', lst[1], flags=re.IGNORECASE).group().lower()
        subset = re.match(r'(dt|et)', lst[2], flags=re.IGNORECASE).group().lower()
        roomtype = re.match(r'(far_room|near_room)\d+', "_".join(lst[-3:-1]), flags=re.IGNORECASE).group().lower()
        room = f'{datatype}_{subset}_{roomtype}'

        spkr_id = lst[0]
        gender_type = REVERB_spk2gender[spkr_id]

        num_correct, num_sub, num_del, num_ins = results[k]

        err = float(num_sub + num_del + num_ins) / (num_correct + num_sub + num_del + num_ins)

        new_dic[gender_type][room].append(err)
        length_dic[len_type][room].append(err)
        if utt2rt60 is not None:
            rt60_dic[rt60_type][room].append(err)
        room_dic[room].append(err)


    for gt in ['F', 'M']:
        for room in [
            'simdata_dt_far_room1', 'simdata_dt_far_room2', 'simdata_dt_far_room3',
            'simdata_dt_near_room1', 'simdata_dt_near_room2', 'simdata_dt_near_room3',
            'simdata_et_far_room1', 'simdata_et_far_room2', 'simdata_et_far_room3',
            'simdata_et_near_room1', 'simdata_et_near_room2', 'simdata_et_near_room3',
            'realdata_dt_far_room1', 'realdata_dt_near_room1',
            'realdata_et_far_room1', 'realdata_et_near_room1',
        ]:
            if len(new_dic[gt][room]) == 0:
                continue
            ret = "{} {}\t".format(gt, room)
            mean_err = sum(new_dic[gt][room]) / len(new_dic[gt][room])
            ret += '{}: {:2.2f} % ({} samples)\t'.format(room, mean_err * 100, len(new_dic[gt][room]))
            print(ret)

    if utt2rt60 is not None:
        print('')
        for rt in ['LowReverb', 'HighReverb']:
            for room in [
                'simdata_dt_far_room1', 'simdata_dt_far_room2', 'simdata_dt_far_room3',
                'simdata_dt_near_room1', 'simdata_dt_near_room2', 'simdata_dt_near_room3',
                'simdata_et_far_room1', 'simdata_et_far_room2', 'simdata_et_far_room3',
                'simdata_et_near_room1', 'simdata_et_near_room2', 'simdata_et_near_room3',
                'realdata_dt_far_room1', 'realdata_dt_near_room1',
                'realdata_et_far_room1', 'realdata_et_near_room1',
            ]:
                if len(rt60_dic[rt][room]) == 0:
                    continue
                ret = "{} {}\t".format(rt, room)
                mean_err = sum(rt60_dic[rt][room]) / len(rt60_dic[rt][room])
                ret += '{}: {:2.2f} % ({} samples)\t'.format(room, mean_err * 100, len(rt60_dic[rt][room]))
                print(ret)

    print('')
    for room in room_dic:
        if len(room_dic[room]) == 0:
            continue
        ret = "{}\t".format(room)
        mean_err = sum(room_dic[room]) / len(room_dic[room])
        ret += '{}: {:2.2f} % ({} samples)\t'.format(room, mean_err * 100, len(room_dic[room]))
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
        for room in [
            'simdata_dt_far_room1', 'simdata_dt_far_room2', 'simdata_dt_far_room3',
            'simdata_dt_near_room1', 'simdata_dt_near_room2', 'simdata_dt_near_room3',
            'simdata_et_far_room1', 'simdata_et_far_room2', 'simdata_et_far_room3',
            'simdata_et_near_room1', 'simdata_et_near_room2', 'simdata_et_near_room3',
            'realdata_dt_far_room1', 'realdata_dt_near_room1',
            'realdata_et_far_room1', 'realdata_et_near_room1',
        ]:
            if len(length_dic[lt][room]) == 0:
                continue
            ret = "{} {}\t".format(lt, room)
            mean_err = sum(length_dic[lt][room]) / len(length_dic[lt][room])
            ret += '{}: {:2.2f}% ({} samples)\t'.format(room, mean_err * 100, len(length_dic[lt][room]))
            print(ret)
