import argparse
from distutils.util import strtobool
from functools import partial
import logging
import os
import yaml

from asteroid_metrics import get_metrics
from asteroid_metrics import average_arrays_in_dic
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import plot_spectrogram
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.transform.transformation import Transformation
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import SoundHDF5File


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def istft(x, n_shift, win_length=None, window='hann', center=True, length=None):
    if x.ndim == 2:
        single_channel = True
        # x: [Time, Freq] -> [Time, Channel, Freq]
        x = x[:, None, :]
    else:
        single_channel = False

    # x: [Time, Channel]
    x = np.stack(
        [
            librosa.istft(
                x[:, ch].T,  # [Time, Freq] -> [Freq, Time]
                hop_length=n_shift,
                win_length=win_length,
                window=window,
                center=center,
                length=length,
            )
            for ch in range(x.shape[1])
        ],
        axis=1,
    )

    if single_channel:
        # x: [Time, Channel] -> [Time]
        x = x[:, 0]
    return x


def filtered_keys(key):
    return not (
        key.startswith('feature_transform.')
        or key.startswith('encoder.')
        or key.startswith('decoder.')
        or key.startswith('ctc.')
    )


def main(args):
    # Get model configuration
    idim, odim, train_args = get_model_conf(args.model_path, None)

    # Initialize inverse_stft
    with open(train_args.preprocess_conf) as file:
        preproc_conf = yaml.load(file, Loader = yaml.FullLoader)
        preproc_conf = preproc_conf['process'][0]

    preprocessing = Transformation(train_args.preprocess_conf)
    preproc = partial(preprocessing, train=False)
    inverse_stft = partial(
        istft,
        n_shift=preproc_conf['n_shift'],
        win_length=preproc_conf['win_length'],
        window=preproc_conf['window']
    )

    # Load model parameters
    E2E = dynamic_import(train_args.model_module)
    model = E2E(idim, odim, train_args)
    del model.feature_transform
    del model.encoder
    del model.decoder
    del model.ctc
    del model.criterion
    snapshot_dict = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    if 'model' in snapshot_dict:
        model.load_state_dict(
            {k: v for k, v in snapshot_dict["model"].items() if filtered_keys(k)}
        )
    else:
        model.load_state_dict(
            {k: v for k, v in snapshot_dict.items() if filtered_keys(k)}
        )
    model.eval()

    # Set hyper-parameters for eavluation
    if hasattr(model.frontend, 'taps'):
        model.frontend.taps = args.test_btaps
        print('setting taps to {}'.format(model.frontend.taps))
    if hasattr(model.frontend, 'btaps'):
        model.frontend.btaps = args.test_btaps
        print('setting btaps to {}'.format(model.frontend.btaps))
    if hasattr(model.frontend, 'wpe') and hasattr(model.frontend.wpe, 'taps'):
        model.frontend.wpe.taps = args.test_btaps
        print('setting wpe.taps to {}'.format(model.frontend.wpe.taps))
    if hasattr(model.frontend, 'beamformer') and hasattr(model.frontend.beamformer, 'btaps'):
        model.frontend.beamformer.btaps = args.test_btaps
        print('setting beamformer.btaps to {}'.format(model.frontend.beamformer.btaps))

    chs = args.test_nmics if args.test_nmics > 0 else 8
    ref_channel = getattr(train_args, "ref_channel", 0)

    # Load evaluation data
    dataset = {}
    with open(os.path.join(args.data_dir, 'wav.scp'), 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) <= 0:
                continue
            utt, wavpath = line.split(maxsplit=1)
            dataset.setdefault(utt, {})["mix"] = wavpath

    with open(os.path.join(args.data_dir, 'spk1.scp'), 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) <= 0:
                continue
            utt, wavpath = line.split(maxsplit=1)
            dataset.setdefault(utt, {})["spk1"] = wavpath

    with open(os.path.join(args.data_dir, 'spk2.scp'), 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) <= 0:
                continue
            utt, wavpath = line.split(maxsplit=1)
            dataset.setdefault(utt, {})["spk2"] = wavpath

    # Prepare output directory for storing enhanced audios
    os.makedirs(args.output_dir, exist_ok=True)

    # Perform evaluation
    compute_metrics = ['si_sdr', 'sdr', 'sir', 'sar', 'stoi', 'pesq']
    eval_results = {metric: [] for metric in compute_metrics}
    eval_results0 = {metric: [] for metric in compute_metrics}

    sample_count = 0
    total_num = len(dataset.keys())
    for utt, wavs in dataset.items():
        sample_count += 1
        logging.warning('(%d/%d) enhanncing ' + utt, sample_count, total_num)

        mixwav, s1wav, s2wav = wavs["mix"], wavs["spk1"], wavs["spk2"]
        wav_mix, sr = sf.read(mixwav)
        wav_mix = wav_mix[:, ref_channel]
        # (2, T)
        wav_ref = np.stack(
            [
#                sf.read(s1wav)[0][:, ref_channel],
#                sf.read(s2wav)[0][:, ref_channel]
                sf.read(s1wav)[0],
                sf.read(s2wav)[0]
            ],
            axis=0
        )
        if wav_ref.ndim == 3:
            wav_ref = wav_ref[..., ref_channel]

        # (1, T, chs)
        xs = preproc(sf.read(mixwav)[0][:, :chs])[None, ...]
        ilens = torch.LongTensor([xs.shape[1]])
        with torch.no_grad():
            h = to_torch_tensor(xs)
            separated, _, predicted_masks = model.frontend(h, ilens)

        length = wav_ref.shape[1]
        # (2, T)
        wav_enh = np.stack(
            [
                inverse_stft(sep[0].numpy(), length=length)
                for sep in separated
            ],
            axis=0
        )

        metrics_dict = get_metrics(
            wav_mix, wav_ref, wav_enh,
            sample_rate=sr,
            metrics_list=compute_metrics,
            average=False,
            compute_permutation=True
        )

        metrics_dict = {k: v.squeeze() for k, v in metrics_dict.items()}
        str_metrics = '\n'.join(['  {}: {}'.format(k, v.tolist()) for k, v in metrics_dict.items()])
        logging.warning(' evaluation results:\n{}'.format(str_metrics))

        avg_metrics_dict = average_arrays_in_dic(metrics_dict)
        for k, v in avg_metrics_dict.items():
            if k.startswith('input_'):
                eval_results0[k[6:]].append(v)
            else:
                eval_results[k].append(v)

        # Save enhanced audios
        sf.write(os.path.join(args.output_dir, utt + '_0.wav'), wav_enh[0], sr)
        sf.write(os.path.join(args.output_dir, utt + '_1.wav'), wav_enh[1], sr)
        if args.plot_masks:
            plot_spectrogram(
                plt,
                predicted_masks[0].detach().numpy()[0, :, ref_channel].T,
                fs=sr,
                mode="linear",
                bottom=False,
                labelbottom=False,
            )
            plt.savefig(os.path.join(args.output_dir, utt + "_mask0.png"))
            plt.clf()
            plot_spectrogram(
                plt,
                predicted_masks[1].detach().numpy()[0, :, ref_channel].T,
                fs=sr,
                mode="linear",
                bottom=False,
                labelbottom=False,
            )
            plt.savefig(os.path.join(args.output_dir, utt + "_mask1.png"))
            plt.clf()


    mean_metrics = {k: np.mean(v) for k, v in eval_results.items()}
    logging.warning('Evaluation of Separated wavs')
    for k, v in eval_results.items():
        logging.warning('mean {}: {}'.format(k.replace('_', '-').upper(), float(np.mean(v))))
    print("\n", flush=True)

    logging.warning('Evaluation of Original Mixture')
    for k, v in eval_results0.items():
        logging.warning('mean {}: {}'.format(k.replace('_', '-').upper(), float(np.mean(v))))
    print("\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='path to the data directory containing wav.scp and ref?.scp')
    parser.add_argument('--model-path', type=str, required=True, help='path to the trained MIMO model')
    parser.add_argument('--output-dir', type=str, required=True, help='output path for storing enhanced wavs')
    parser.add_argument('--test-btaps', type=int, default=-1, help='set number of filter taps (length) during testing, same as training by default')
    parser.add_argument('--test-nmics', type=int, default=-1, help='set number of microphones during testing, same as training by default')
    parser.add_argument('--plot-masks', type=str2bool, default=False, help='True to plot predicted masks')
    args = parser.parse_args()
    main(args)
