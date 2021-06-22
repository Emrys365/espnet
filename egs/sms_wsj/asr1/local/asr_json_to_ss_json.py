#!/usr/bin/env python3


import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asr-json', type=str,
                        help='data.json for ASR')
    parser.add_argument('--wav-scp', type=str, required=True,
                        help='wav.scp for speech separation, containing paths of mixed speech and individual speech of each speaker')
    parser.add_argument('--spk1-scp', type=str, required=True,
                        help='spk1.scp for speech separation, containing paths of speech of speaker 1')
    parser.add_argument('--spk2-scp', type=str, required=True,
                        help='spk2.scp for speech separation, containing paths of speech of speaker 2')
    args = parser.parse_args()

    with open(args.asr_json, 'r') as f:
        json_asr = json.load(f)['utts']

    wavs = {}
    with open(args.wav_scp, 'r') as f:
        for line in f:
            if line.strip():
                k, v = line.strip().split(maxsplit=1)
                wavs[k] = {'mix': v}
    with open(args.spk1_scp, 'r') as f:
        for line in f:
            if line.strip():
                k, v = line.strip().split(maxsplit=1)
                wavs[k]['spk1'] = v
    with open(args.spk2_scp, 'r') as f:
        for line in f:
            if line.strip():
                k, v = line.strip().split(maxsplit=1)
                wavs[k]['spk2'] = v

    for uttid, info in wavs.items():
        wav_mix, wav_spk1, wav_spk2 = info.values()
        json_asr[uttid]['input'][0]['feat'] = wav_mix
        json_asr[uttid]['input'][0]['filetype'] = 'sound'
        json_asr[uttid]['speaker1'] = {
            'filetype': 'sound',
            'input_feat': wav_spk1,
        }
        json_asr[uttid]['speaker2'] = {
            'filetype': 'sound',
            'input_feat': wav_spk2,
        }

    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps({'utts': json_asr}, indent=4, ensure_ascii=False, sort_keys=True)
    print(jsonstring)
