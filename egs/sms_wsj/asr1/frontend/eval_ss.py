#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from collections import OrderedDict
from distutils.util import strtobool
import itertools
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import soundfile
import mir_eval
import pypesq
from pystoi.stoi import stoi

from espnet.utils.cli_utils import get_commandline_args


PY2 = sys.version_info[0] == 2


def eval_STOI(ref, y, fs, extended=False, compute_permutation=True):
    """Calculate STOI

    Reference:
        A short-time objective intelligibility measure
            for time-frequency weighted noisy speech
        https://ieeexplore.ieee.org/document/5495701

    Note(kamo):
        STOI is defined on the signal at 10kHz
        and the input at the other sampling rate will be resampled.
        Thus, the result differs depending on the implementation of resampling.
        Especially, pystoi cannot reproduce matlab's resampling now.

    :param ref (np.ndarray): Reference (Nsrc, Nframe, Nmic) or (Nsrc, Nframe)
    :param y (np.ndarray): Enhanced (Nsrc, Nframe, Nmic) or (Nsrc, Nframe)
    :param fs (int): Sample frequency
    :param extended (bool): stoi or estoi
    :param compute_permutation (bool):
    :return: value, perm
    :rtype: Tuple[Tuple[float, ...], Tuple[int, ...]]
    """
    if ref.shape != y.shape:
        raise ValueError('ref and y should have the same shape: {} != {}'
                         .format(ref.shape, y.shape))
    if ref.ndim not in (2, 3):
        raise ValueError('Input must have 2 or 3 dims: {}'.format_map(ref.ndim))
    n_src = ref.shape[0]
    n_mic = ref.shape[2] if ref.ndim == 3 else 0

    if compute_permutation:
        index_list = list(itertools.permutations(range(n_src)))
    else:
        index_list = [list(range(n_src))]

    if n_mic == 0:
        values = [[stoi(ref[i, :], y[j, :], fs, extended)
                   for i, j in enumerate(indices)]
                  for indices in index_list]
    else:
        values = [[sum(stoi(ref[i, :, ch], y[j, :, ch], fs, extended)
                    for ch in range(n_mic)) / n_mic
                   for i, j in enumerate(indices)]
                  for indices in index_list]

    best_pairs = sorted([(v, i) for v, i in zip(values, index_list)],
                        key=lambda x: sum(x[0]))[-1]
    value, perm = best_pairs
    return tuple(value), tuple(perm)


def eval_PESQ(ref, y, fs, compute_permutation=True):
    """Evaluate PESQ

    PESQ program can be downloaded from here:
        https://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200102-I!!SOFT-ZST-E&type=items

    Reference:
        Perceptual evaluation of speech quality (PESQ)-a new method
            for speech quality assessment of telephone networks and codecs
        https://ieeexplore.ieee.org/document/941023

    :param x (np.ndarray): Reference (Nsrc, Nframe, Nmic) or (Nsrc, Nframe)
    :param y (np.ndarray): Enhanced (Nsrc, Nframe, Nmic) or (Nsrc, Nframe)
    :param fs (int): Sample frequency
    :param compute_permutation (bool):
    """
    if fs not in (8000, 16000):
        raise ValueError('Sample frequency must be 8000 or 16000: {}'
                         .format(fs))
    if ref.shape != y.shape:
        raise ValueError('ref and y should have the same shape: {} != {}'
                         .format(ref.shape, y.shape))
    if ref.ndim not in (2, 3):
        raise ValueError('Input must have 2 or 3 dims: {}'.format_map(ref.ndim))
    n_src = ref.shape[0]
    n_mic = ref.shape[2] if ref.ndim == 3 else 0

    if compute_permutation:
        index_list = list(itertools.permutations(range(n_src)))
    else:
        index_list = [list(range(n_src))]

    if n_mic == 0:
        values = [[pypesq.pesq(ref[i, :], y[j, :], fs)
                   for i, j in enumerate(indices)]
                  for indices in index_list]
    else:
        values = [[sum(pypesq.pesq(ref[i, :, ch], y[j, :, ch], fs)
                    for ch in range(n_mic)) / n_mic
                   for i, j in enumerate(indices)]
                  for indices in index_list]

    best_pairs = sorted([(v, i) for v, i in zip(values, index_list)],
                        key=lambda x: sum(x[0]))[-1]
    value, perm = best_pairs
    return tuple(value), tuple(perm)


def si_sdr(reference, estimation, scaling=True):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
        scaling: bool
    Returns:
        SI-SDR
    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    >>> np.random.seed(0)
    >>> reference = np.random.randn(100)
    >>> si_sdr(reference, reference)
    inf
    >>> si_sdr(reference, reference * 2)
    inf
    >>> si_sdr(reference, np.flip(reference))
    -25.127672346460717
    >>> si_sdr(reference, reference + np.flip(reference))
    0.481070445785553
    >>> si_sdr(reference, reference + 0.5)
    6.3704606032577304
    >>> si_sdr(reference, reference * 2 + 1)
    6.3704606032577304
    >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
    nan
    >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
    array([6.3704606, 6.3704606])
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)

    # assert reference.dtype == np.float64, reference.dtype
    # assert estimation.dtype == np.float64, estimation.dtype

    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    if scaling:
        optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
            / reference_energy
    else:
        optimal_scaling = 1

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)


def eval_SI_measures(ref, y, scaling=True, compute_permutation=True):
    """Evaluate scale-invariant SDR, SIR and SAR (SI-SDR, SI-SIR, SI-SAR)

    Reference:
        SDR - half-baked or well done?
            by Jonathan Le Roux, Scott Wisdom, Hakan Erdogan, John R. Hershey
        https://arxiv.org/abs/1811.02508

    :param x (np.ndarray): Reference (Nsrc, Nframe, Nmic) or (Nsrc, Nframe)
    :param y (np.ndarray): Enhanced (Nsrc, Nframe, Nmic) or (Nsrc, Nframe)
    :param scaling (bool):
    :param compute_permutation (bool):
    """
    if ref.shape != y.shape:
        raise ValueError('ref and y should have the same shape: {} != {}'
                         .format(ref.shape, y.shape))
    if ref.ndim not in (2, 3):
        raise ValueError('Input must have 2 or 3 dims: {}'.format_map(ref.ndim))
    n_src = ref.shape[0]
    n_mic = ref.shape[2] if ref.ndim == 3 else 0

    if compute_permutation:
        index_list = list(itertools.permutations(range(n_src)))
    else:
        index_list = [list(range(n_src))]

    if n_mic == 0:
        values = [[si_sdr(ref[i, :], y[j, :], scaling=scaling)
                   for i, j in enumerate(indices)]
                  for indices in index_list]
    else:
        values = [[sum(si_sdr(ref[i, :, ch], y[j, :, ch], scaling=scaling)
                    for ch in range(n_mic)) / n_mic
                   for i, j in enumerate(indices)]
                  for indices in index_list]

    best_pairs = sorted([(v, i) for v, i in zip(values, index_list)],
                        key=lambda x: sum(x[0]))[-1]
    value, perm = best_pairs
    return tuple(value), tuple(perm)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate enhanced speech. '
                    'e.g. {c} --ref ref.scp --enh enh.scp --outdir outputdir'
                    'or {c} --ref ref.scp ref2.scp --enh enh.scp enh2.scp '
                    '--outdir outputdir'
                    .format(c=sys.argv[0]),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--ref', dest='reffiles', nargs='+', type=str,
                        required=True,
                        help='WAV file lists for reference')
    parser.add_argument('--enh', dest='enhfiles', nargs='+', type=str,
                        required=True,
                        help='WAV files lists for enhanced')
    parser.add_argument('--outdir', type=str,
                        required=True)
    parser.add_argument('--keylist', type=str,
                        help='Specify the target samples. By default, '
                             'using all keys in the first reference file')
    parser.add_argument('--evaltypes', type=str, nargs='+',
                        choices=['SDR', 'STOI', 'ESTOI', 'PESQ'],
                        default=['SDR', 'PESQ'])
    parser.add_argument('--permutation', type=strtobool, default=True,
                        help='Compute all permutations or '
                             'use the pair of input order')

    # About BSS Eval v4:
    # The 2018 Signal Separation Evaluation Campaign
    # https://arxiv.org/abs/1804.06267
    parser.add_argument('--bss-eval-images', type=strtobool, default=True,
                        help='Use bss_eval_images or bss_eval_sources. '
                             'For more detail, see museval source codes.')
    parser.add_argument('--bss-eval-version', type=str,
                        default='v3', choices=['v3', 'v4'],
                        help='Specify bss-eval-version: v3 or v4')
    args = parser.parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())
    if len(args.reffiles) != len(args.enhfiles):
        raise RuntimeError(
            'The number of ref files are different '
            'from the enh files: {} != {}'.format(len(args.reffiles),
                                                  len(args.enhfiles)))
    if len(args.enhfiles) == 1:
        args.permutation = False

    # Read text files and created a mapping of key2filepath
    reffiles_dict = OrderedDict()  # Dict[str, Dict[str, str]]
    for ref in args.reffiles:
        d = OrderedDict()
        with open(ref, 'r') as f:
            for line in f:
                key, path = line.split(None, 1)
                d[key] = path.rstrip()
        reffiles_dict[ref] = d

    enhfiles_dict = OrderedDict()  # Dict[str, Dict[str, str]]
    for enh in args.enhfiles:
        d = OrderedDict()
        with open(enh, 'r') as f:
            for line in f:
                key, path = line.split(None, 1)
                d[key] = path.rstrip()
        enhfiles_dict[enh] = d

    if args.keylist is not None:
        with open(args.keylist, 'r') as f:
            keylist = [line.rstrip().split()[0] for line in f]
    else:
        keylist = list(reffiles_dict.values())[0]

    if len(keylist) == 0:
        raise RuntimeError('No keys are found')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    evaltypes = []
    for evaltype in args.evaltypes:
        if evaltype == 'SDR':
            evaltypes += ['SI-SDR', 'SDR', 'ISR', 'SIR', 'SAR']
        else:
            evaltypes.append(evaltype)

    # Open files in write mode
    writers = {k: open(os.path.join(args.outdir, k), 'w') for k in evaltypes}

    for key in keylist:
        # 1. Load ref files
        rate_prev = None

        ref_signals = []
        for listname, d in reffiles_dict.items():
            if key not in d:
                raise RuntimeError('{} doesn\'t exist in {}'
                                   .format(key, listname))
            filepath = d[key]
            signal, rate = soundfile.read(filepath, dtype=np.int16)
            if signal.ndim == 1:
                # (Nframe) -> (Nframe, 1)
                signal = signal[:, None]
            ref_signals.append(signal)
            if rate_prev is not None and rate != rate_prev:
                raise RuntimeError('Sampling rates mismatch')
            rate_prev = rate

        # 2. Load enh files
        enh_signals = []
        for listname, d in enhfiles_dict.items():
            if key not in d:
                raise RuntimeError('{} doesn\'t exist in {}'
                                   .format(key, listname))
            filepath = d[key]
            signal, rate = soundfile.read(filepath, dtype=np.int16)
            if signal.ndim == 1:
                # (Nframe) -> (Nframe, 1)
                signal = signal[:, None]
            enh_signals.append(signal)
            if rate_prev is not None and rate != rate_prev:
                raise RuntimeError('Sampling rates mismatch')
            rate_prev = rate

        for signal in ref_signals + enh_signals:
            if signal.shape[1] != ref_signals[0].shape[1]:
                raise RuntimeError('The number of channels mismatch')

        # 3. Zero padding to adjust the length to the maximum length in inputs
        ml = max(len(s) for s in ref_signals + enh_signals)
        ref_signals = [np.pad(s, [(0, ml - len(s)), (0, 0)], mode='constant')
                       if len(s) < ml else s for s in ref_signals]

        enh_signals = [np.pad(s, [(0, ml - len(s)), (0, 0)], mode='constant')
                       if len(s) < ml else s for s in enh_signals]

        # ref_signals, enh_signals: (Nsrc, Nframe, Nmic)
        ref_signals = np.stack(ref_signals, axis=0)
        enh_signals = np.stack(enh_signals, axis=0)

        # 4. Evaluates
        for evaltype in args.evaltypes:
            if evaltype == 'SI-SDR':
                ref_signals = ref_signals.squeeze(-1)
                enh_signals = enh_signals.squeeze(-1)
                (si_sdr, perm) = \
                    eval_SI_measures(ref_signals, enh_signals, scaling=True, compute_permutation=True)

                # sdr: (Nsrc, Nframe)
                writers['SI-SDR'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, si_sdr))))

            if evaltype == 'SDR':
                ref_signals = ref_signals.squeeze(-1)
                enh_signals = enh_signals.squeeze(-1)
                (sdr, sir, sar, perm) = \
                    mir_eval.separation.bss_eval_sources(
                        ref_signals, enh_signals,
                        compute_permutation=True)

                # sdr: (Nsrc, Nframe)
                writers['SDR'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, sdr))))
                writers['SIR'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, sir))))
                writers['SAR'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, sar))))

            elif evaltype == 'STOI':
                stoi, perm = eval_STOI(ref_signals, enh_signals, rate,
                                       extended=False,
                                       compute_permutation=args.permutation)
                writers['STOI'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, stoi))))

            elif evaltype == 'ESTOI':
                estoi, perm = eval_STOI(ref_signals, enh_signals, rate,
                                        extended=True,
                                        compute_permutation=args.permutation)
                writers['ESTOI'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, estoi))))

            elif evaltype == 'PESQ':
                pesq, perm = eval_PESQ(ref_signals, enh_signals, rate,
                                       compute_permutation=args.permutation)
                writers['PESQ'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, pesq))))
            else:
                # Cannot reach
                raise RuntimeError


if __name__ == "__main__":
    main()
