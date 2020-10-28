#!/usr/bin/env python3
# Generate single-speaker source speech data from the spatialized mixed speech data

import argparse
import os


def main(args):
    reco2wav = {}
    reco2text = {}
    utt2spk = {}
    # for spk in range(args.num_spkrs):
    with open(os.path.join(args.datadir, 'ref{}.scp'.format(spk + 1))) as f:
        for line in f:
            if len(line.strip()) <= 0:
                continue
            uttid, wavpath = line.strip().split(maxsplit=1)
            # uttid: 447_445_447c0407_-0.58435_445c040q_0.58435_reverb


    os.makedirs(args.outputdir, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='path to the data directory of the mixed speech')
    parser.add_argument('--outputdir', type=str, required=True, help='path to the directory for storing the generated single-speaker data')
    # parser.add_argument('--num-spkrs', type=int, default=2, help='maximum number of speakers in the mixed speech')
    args = parser.parse_args()
    main(args)
