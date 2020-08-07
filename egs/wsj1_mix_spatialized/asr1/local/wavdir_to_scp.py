#!/usr/bin/env python3

from pathlib import Path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('mix_wav_dir', type=str, help='directory of mixed speech')
parser.add_argument('--s1-wav-dir', type=str, default='', help='directory of speech of speaker 1')
parser.add_argument('--s2-wav-dir', type=str, default='', help='directory of speech of speaker 2')
parser.add_argument('--output-scp', type=str, default='wavdata.scp', help='path of the output scp file')
args = parser.parse_args()

# Assume that all three directories share the same wav file names
s1_wav_dir = Path(args.s1_wav_dir).resolve() if len(args.s1_wav_dir) > 0 else None
s2_wav_dir = Path(args.s2_wav_dir).resolve() if len(args.s2_wav_dir) > 0 else None
assert (s1_wav_dir == s2_wav_dir == None) or (s1_wav_dir.exists() and s2_wav_dir.exists())

mix_wav_dir = Path(args.mix_wav_dir)
if mix_wav_dir.exists():
    wav_list = mix_wav_dir.glob('*.wav')
    # search for all wav files in the directory recursively
    # wav_list = list(mix_wav_dir.rglob('*.wav'))
else:
    raise ValueError('No such file or directory: {}'.format(args.mix_wav_dir))

scp_list = []
for line in wav_list:
    # get absolute path of the wav file, resolving all symlinks on the way and also normalizing it
    mix_wav = line.resolve()
    scp_line = str(mix_wav)
    basename = mix_wav.name
    if s1_wav_dir:
        s1_wav = s1_wav_dir.joinpath(basename)
        s2_wav = s2_wav_dir.joinpath(basename)
        scp_line += ' {} {}'.format(str(s1_wav), str(s2_wav))
    scp_list.append(scp_line)

with open(args.output_scp, 'w') as f:
    for line in scp_list:
        f.write(line + '\n')