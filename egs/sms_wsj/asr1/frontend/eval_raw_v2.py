import argparse
from distutils.util import strtobool
from itertools import chain
import json
import logging
import os
import yaml

from asteroid_metrics import average_arrays_in_dic
from asteroid_metrics import get_metrics
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from torch_complex.tensor import ComplexTensor

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import plot_spectrogram
from espnet.nets.pytorch_backend.e2e_asr_mix_transformer_ss import _create_mask_label
from espnet.nets.pytorch_backend.frontends.dnn_wpe import DNN_WPE
from espnet.nets.pytorch_backend.frontends.stft import Stft
from espnet.utils.dynamic_import import dynamic_import


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


@torch.no_grad()
def main(args):
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s %(filename)s %(levelname)s] %(message)s',
        datefmt='%d %b %Y %H:%M:%S',
    )

    default_ref_channel = 0
    # Get model configuration
    idim, odim, train_args = get_model_conf(args.model_path, None)

    # Initialize inverse_stft
    with open(train_args.preprocess_conf) as file:
        preproc_conf = yaml.load(file, Loader=yaml.FullLoader)
        preproc_conf = preproc_conf["process"][0]
    stft = Stft(
        win_length=preproc_conf["win_length"],
        n_fft=preproc_conf["n_fft"],
        hop_length=preproc_conf["n_shift"],
        window=preproc_conf["window"],
    )

    # Load model parameters
    E2E = dynamic_import(train_args.model_module)
    model = E2E(idim, odim, train_args)
    num_spkrs = model.num_spkrs
    for attr in ("feature_transform", "encoder", "decoder", "ctc", "criterion"):
        delattr(model, attr)
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
    model.to(device=args.device)

    # Set hyper-parameters for evaluation
    if args.force_using_wpe and not getattr(model.frontend, 'wpe', None):
        print('Force using Nara-WPE')
        model.frontend.use_wpe = True
        model.frontend.wpe = DNN_WPE(use_dnn_mask=False, iterations=3)
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
    ref_channel = getattr(train_args, "ref_channel", default_ref_channel)

    if args.use_oracle_mask:
        print('Using oracle masks ({}) for WPE and beamforming'.format(args.mask_type))
    if args.iterative_update > 0:
        print("Performing iterative update for WPE+beamforming (%d-iter)" % args.iterative_update)

    # Load evaluation data
    dataset = {}
    with open(os.path.join(args.data_dir, 'wav' + args.wav_scp_suffix), 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) <= 0:
                continue
            utt, wavpath = line.split(maxsplit=1)
            dataset.setdefault(utt, {})["mix"] = wavpath

    for spk in range(model.num_spkrs):
        with open(os.path.join(args.data_dir, f'spk{spk + 1}' + args.wav_scp_suffix), 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) <= 0:
                    continue
                utt, wavpath = line.split(maxsplit=1)
                dataset.setdefault(utt, {})[f"spk{spk + 1}"] = wavpath

    if os.path.exists(os.path.join(args.data_dir, 'noise1' + args.wav_scp_suffix)):
        with open(os.path.join(args.data_dir, 'noise1' + args.wav_scp_suffix), 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) <= 0:
                    continue
                utt, wavpath = line.split(maxsplit=1)
                dataset.setdefault(utt, {})["noise"] = wavpath

    if args.resolve_freq_perm:
        print("Resolving frequency permutation problem via DOA estimation")
        assert os.path.exists(args.sensor_pos_json), args.sensor_pos_json
        with open(args.sensor_pos_json, "r") as f:
            sensor_pos_info = json.load(f)

    if args.output_dir:
        # Prepare output directory for storing enhanced audios
        os.makedirs(args.output_dir, exist_ok=True)

    if args.write_scps:
        assert args.write_scp_dir is not None
        os.makedirs(args.write_scp_dir, exist_ok=True)

    # Perform evaluation
    compute_metrics = ['si_sdr', 'sdr', 'sir', 'sar', 'stoi', 'pesq', 'srmr']
    eval_results = {metric: [] for metric in compute_metrics}
    eval_results0 = {metric: [] for metric in compute_metrics}
    # for 1-pass mask
    eval_results1 = {metric: [] for metric in compute_metrics}
    # for multi-iter mask
    eval_results2 = {metric: [] for metric in compute_metrics}

    sample_count = 0
    total_num = len(dataset.keys())
    for utt, wavs in dataset.items():
        sample_count += 1
        logging.warning('(%d/%d) enhanncing ' + utt, sample_count, total_num)

        mixwav = wavs["mix"]
        spk_wav = [wavs[f"spk{spk + 1}"] for spk in range(model.num_spkrs)]
        wav_mix0, sr = sf.read(mixwav)
        wav_mix = wav_mix0[:, ref_channel]
        # (2, T)
        wav_ref0 = np.stack(
            [
#                sf.read(s1wav)[0][:, ref_channel],
#                sf.read(s2wav)[0][:, ref_channel]
#                sf.read(s1wav)[0],
#                sf.read(s2wav)[0]
                sf.read(swav)[0] for swav in spk_wav
            ],
            axis=0
        )
        if wav_ref0.ndim == 3:
            wav_ref = wav_ref0[..., ref_channel]
        else:
            wav_ref = wav_ref0

        if wav_ref.shape[1] > wav_mix.shape[0]:
            print("[WARNING] clipping long reference to match the length of input wav", flush=True)
            wav_ref = wav_ref[:, :wav_mix.shape[0]]
        elif wav_ref.shape[1] < wav_mix.shape[0]:
            print("[WARNING] clipping long input wav to match the length of reference", flush=True)
            wav_mix = wav_mix[:wav_ref.shape[1]]

        # (1, T, chs)
        xs = torch.as_tensor(wav_mix0[None, :wav_mix.shape[0], :chs], device=args.device, dtype=torch.float32)
        speech_lengths = torch.LongTensor([wav_mix.shape[0]], device=args.device)
        # (1, T', C, F)
        xs = ComplexTensor(*torch.unbind(stft(xs, speech_lengths)[0], dim=-1))

        if args.use_oracle_mask:
            noisewav = wavs.get("noise", None)
            if noisewav is not None:
                wav_noise = sf.read(noisewav)[0][:wav_mix.shape[1]]
                noise = torch.as_tensor(wav_noise[None, :, :chs], device=args.device, dtype=torch.float32)
                noise_spec = ComplexTensor(*torch.unbind(stft(noise, speech_lengths)[0], dim=-1))

            if wav_ref0.ndim == 3:
                ys = torch.as_tensor(wav_ref0[:, None, :wav_mix.shape[0], :chs], device=args.device, dtype=torch.float32)
            else:
                ys = torch.as_tensor(wav_ref0[:, None, :wav_mix.shape[0], None], device=args.device, dtype=torch.float32)

            ys = [ComplexTensor(*torch.unbind(stft(y, speech_lengths)[0], dim=-1)) for y in ys]
            # [(B, F, C, T')]
            mask_speech = [m.permute(0, 3, 2, 1) for m in _create_mask_label(xs, ys, mask_type=args.mask_type)]

            if noisewav is not None:
                mask_noise = [m.permute(0, 3, 2, 1) for m in _create_mask_label(noise_spec, ys, mask_type=args.mask_type)]
            else:
                mask_noise = [0 for _ in mask_speech]
            for spk, mask_n in enumerate(mask_noise):
                mask_s = mask_speech.pop(spk)
                mask_noise[spk] = mask_n + sum(mask_speech)
                mask_speech.insert(spk, mask_s)
            #masks = mask_speech + list(chain.from_iterable(zip(mask_speech, mask_noise)))
            masks = mask_speech + mask_speech + mask_noise
        else:
            masks = None

        ilens = torch.LongTensor([xs.shape[1]], device=args.device)
        separated, _, predicted_masks = model.frontend(xs, ilens, masks=masks)
        if args.resolve_freq_perm:
            separated = model.frontend._resolve_frequency_permutation(
                xs, separated, sensor_pos_info[utt],
                fs=sr,
                freq_min=400,
                freq_max=4000,
                resolution=1.0,
                sound_velocity=343,
                threshold=args.freq_perm_thres,
            )
        if model.num_spkrs == 1:
            separated = [separated]

        #######################################
        ### added for masking based separation
#        wav_sep1 = np.stack(
#            [
#                stft.inverse(
#                    xs[..., ref_channel, :] * m[..., ref_channel, :],
#                    speech_lengths
#                )[0].squeeze(0).cpu().numpy()
#                for m in predicted_masks[model.num_spkrs:][::2]
#            ],
#            axis=0
#        )
#        metrics_dict1 = get_metrics(
#            wav_mix, wav_ref, wav_sep1,
#            sample_rate=sr,
#            metrics_list=compute_metrics,
#            average=False,
#            compute_permutation=True
#        )
#        metrics_dict1 = {k: v.squeeze() for k, v in metrics_dict1.items()}
#
#        if args.verbose:
#            str_metrics1 = '\n'.join(['  {}: {}'.format(k, v.tolist()) for k, v in metrics_dict1.items()])
#            logging.warning(' 1-pass masking evaluation results:\n{}'.format(str_metrics1))
#
#        avg_metrics_dict1 = average_arrays_in_dic(metrics_dict1)
#        for k, v in avg_metrics_dict1.items():
#            if not k.startswith('input_'):
#                eval_results1[k].append((utt, v))
        #######################################

        if args.iterative_update > 0:
            mask_noise = [m.transpose(-1, -3) for m in predicted_masks[model.num_spkrs:][1::2]]
            for _ in range(args.iterative_update):
                separated = [sep.unsqueeze(-2) for sep in separated]
                mask_speech = [m.permute(0, 3, 2, 1) for m in _create_mask_label(xs[..., ref_channel:ref_channel+1, :], separated, mask_type=args.mask_type)]
                masks = mask_speech + mask_speech + mask_noise #[1 - m for m in mask_speech]
                separated, _, predicted_masks = model.frontend(xs, ilens, masks=masks)
                if args.resolve_freq_perm:
                    separated = model.frontend._resolve_frequency_permutation(
                        xs, separated, sensor_pos_info[utt],
                        fs=sr,
                        freq_min=400,
                        freq_max=4000,
                        resolution=1.0,
                        sound_velocity=343,
                        threshold=args.freq_perm_thres,
                    )
                if model.num_spkrs == 1:
                    separated = [separated]


            ### added for masking based separation
            refch = 0 if predicted_masks[0].shape[-2] == 1 else ref_channel
            wav_sep2 = np.stack(
                [
                    stft.inverse(
                        xs[..., ref_channel, :] * m[..., refch, :],
                        speech_lengths
                    )[0].squeeze(0).cpu().numpy()
                    for m in predicted_masks[model.num_spkrs:][::2]
                ],
                axis=0
            )
            metrics_dict2 = get_metrics(
                wav_mix, wav_ref, wav_sep2,
                sample_rate=sr,
                metrics_list=compute_metrics,
                average=False,
                compute_permutation=True
            )
            metrics_dict2 = {k: v.squeeze() for k, v in metrics_dict2.items()}
            if args.verbose:
                str_metrics2 = '\n'.join(['  {}: {}'.format(k, v.tolist()) for k, v in metrics_dict2.items()])
                logging.warning(' {}-iter masking evaluation results:\n{}'.format(args.iterative_update, str_metrics2))

            avg_metrics_dict2 = average_arrays_in_dic(metrics_dict2)
            for k, v in avg_metrics_dict2.items():
                if not k.startswith('input_'):
                    eval_results2[k].append((utt, v))
        #######################################

        length = wav_ref.shape[1]
        # (2, T)
        wav_enh = np.stack(
            [
                stft.inverse(sep, speech_lengths)[0].squeeze(0).cpu().numpy()
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
        if args.verbose:
            str_metrics = '\n'.join(['  {}: {}'.format(k, v.tolist()) for k, v in metrics_dict.items()])
            logging.warning(' evaluation results:\n{}'.format(str_metrics))

        avg_metrics_dict = average_arrays_in_dic(metrics_dict)
        for k, v in avg_metrics_dict.items():
            if k.startswith('input_'):
                eval_results0[k[6:]].append((utt, v))
            else:
                eval_results[k].append((utt, v))

        if args.output_dir:
            # Save enhanced audios
            for spk in range(model.num_spkrs):
                sf.write(os.path.join(args.output_dir, utt + f'_{spk}.wav'), wav_enh[spk], sr)
            if args.plot_masks:
                for spk in range(model.num_spkrs):
                    plot_spectrogram(
                        plt,
                        predicted_masks[spk].detach().numpy()[0, :, ref_channel].T,
                        fs=sr,
                        mode="linear",
                        bottom=False,
                        labelbottom=False,
                    )
                    plt.savefig(os.path.join(args.output_dir, utt + f"_mask{spk}.png"))
                    plt.clf()

    # mean_metrics = {k: np.mean(v) for k, v in eval_results.items()}
    logging.warning('Evaluation of Separated wavs')
    for k, vs in eval_results.items():
        v = list(zip(*vs))[1]
        logging.warning('mean {}: {}'.format(k.replace('_', '-').upper(), float(np.mean(v))))
    print("\n", flush=True)

#   logging.warning('Evaluation of (1-pass) Masked Mixture')
#   for k, vs in eval_results1.items():
#       v = list(zip(*vs))[1]
#       logging.warning('mean {}: {}'.format(k.replace('_', '-').upper(), float(np.mean(v))))
#   print("\n", flush=True)

    if args.iterative_update > 0:
        logging.warning('Evaluation of ({}-iter) Masked Mixture'.format(args.iterative_update))
        for k, vs in eval_results2.items():
            v = list(zip(*vs))[1]
            logging.warning('mean {}: {}'.format(k.replace('_', '-').upper(), float(np.mean(v))))
        print("\n", flush=True)

    logging.warning('Evaluation of Original Mixture')
    for k, vs in eval_results0.items():
        v = list(zip(*vs))[1]
        logging.warning('mean {}: {}'.format(k.replace('_', '-').upper(), float(np.mean(v))))
    print("\n", flush=True)

    if args.write_scps:
        for metric, vs in eval_results.items():
            # utts, scores = list(zip(*vs))
            with open(os.path.join(args.write_scp_dir, metric + args.write_scp_suffix), "w") as f:
                for utt, score in vs:
                    f.write(f"{utt} {score}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='path to the data directory containing wav.scp and ref?.scp')
    parser.add_argument('--model-path', type=str, required=True, help='path to the trained MIMO model')
    parser.add_argument("--device", type=str, default="cpu", help='"cpu" or "cuda"')
    parser.add_argument('--output-dir', type=str, default=None, help='output path for storing enhanced wavs')
    parser.add_argument("--force-using-wpe", type=str2bool, default=False, help="True to apply Nara-WPE as preprocessing")
    parser.add_argument('--test-btaps', type=int, default=-1, help='set number of filter taps (length) during testing, same as training by default')
    parser.add_argument('--test-nmics', type=int, default=-1, help='set number of microphones during testing, same as training by default')
    parser.add_argument("--use-oracle-mask", type=str2bool, default=False, help="Whether to use oracle masks instead of NN estimated masks")
    parser.add_argument("--mask-type", type=str, default="PSM", help="Type of reference masks for beamforming")
    parser.add_argument('--plot-masks', type=str2bool, default=False, help='True to plot predicted masks')
    parser.add_argument("--iterative-update", type=int, default=0, help="Number of iterations for iterative updating WPE and beamforming results")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Whether to print detailed logs for every sample")
    parser.add_argument("--wav-scp-suffix", type=str, default=".scp", help="Suffix of the scp files to be read")
    parser.add_argument("--write-scps", type=str2bool, default=False, help="Whether to write evaluation results in scp files")
    parser.add_argument("--write-scp-dir", type=str, default=None, help="Directory of the scp file to be written")
    parser.add_argument("--write-scp-suffix", type=str, default=".scp", help="Suffix of the scp file to be written")

    parser.add_argument("--resolve-freq-perm", type=str2bool, default=False, help="Whether to resolve the frequency permutation problem via DOA estimation")
    parser.add_argument("--freq-perm-thres", type=float, default=180.0, help="Threshold used when resolving the frequency permutation problem via DOA estimation")
    parser.add_argument("--sensor-pos-json", type=str, default="", help="Path to the json file containing sensor position information for each sample")
    args = parser.parse_args()
    main(args)
