#!/usr/bin/env python
# Copyright 2018 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Apache 2.0
# Works with both python2 and python3
# This script assumes that WPE (nara_wpe) is installed locally using miniconda.
# ../../../tools/extras/install_miniconda.sh and ../../../tools/extras/install_wpe.sh
# needs to be run and this script needs to be launched run with that version of
# python.
# See local/run_wpe.sh for example.

import numpy as np
import soundfile as sf
import time
import os, errno
from tqdm import tqdm
import argparse

from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft
from nara_wpe import project_root

parser = argparse.ArgumentParser()
parser.add_argument('--files', '-f', nargs='+')
parser.add_argument('--channels', '-c', type=int, default=2)
parser.add_argument('--iterations', '-i', type=int, default=5)
args = parser.parse_args()

#input_files = args.files[:len(args.files)//2]
#output_files = args.files[len(args.files)//2:]
input_files = args.files[0]
output_files = args.files[1:]
out_dir = os.path.dirname(output_files[0])
try:
    os.makedirs(out_dir)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

stft_options = dict(
    size=512,
    shift=128,
    window_length=None,
    fading=True,
    pad=True,
    symmetric_window=False
)

sampling_rate = 16000
delay = 3
iterations = args.iterations # 5 by default
taps = 10

#signal_list = [
#    sf.read(f)[0]
#    for f in input_files
#]
dir_list = input_files.split('/')
if 'tr' in dir_list:
    setname = 'tr'
elif 'cv' in dir_list:
    setname = 'cv'
elif 'tt' in dir_list:
    setname = 'tt'
else:
    raise ValueError('Invalid dataset in {}'.format(input_files))

input_files2 = '/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.2/multi-channel/wsj-mix-spatialized/spatialized_2ch_wav/2speaker_reverb/{}/{}'.format(setname, os.path.basename(input_files))
signal_list = sf.read(input_files2, always_2d=True)[0]
y = signal_list[:, :args.channels].transpose()
Y = stft(y, **stft_options).transpose(2, 0, 1)
Z = wpe(Y, iterations=iterations, statistics_mode='full').transpose(1, 2, 0)
z = istft(Z, size=stft_options['size'], shift=stft_options['shift'])

for d in range(y.shape[0]):
    sf.write(output_files[d], z[d,:], sampling_rate)
