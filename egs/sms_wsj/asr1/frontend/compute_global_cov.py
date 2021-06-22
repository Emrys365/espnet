#!/usr/bin/env python3

from espnet.nets.pytorch_backend.frontends.WPD_beamfomer_v6 import get_covariances
from data_io import LoadInputsAndTargets
import json
from os.path import join
import torch
from torch_complex.tensor import ComplexTensor


def compute_global_statistics(args):
    """Compute gloabl spatial covariance statistics of the reverberant adn anechoic speech.

    Args:
        args (namespace): The program arguments.

    """
    # read json data
    with open(args.train_json, 'rb') as f:
        js = json.load(f)['utts']
    total_num = len(js.keys())

    load_inputs_and_targets = LoadInputsAndTargets(
        load_output=True, sort_in_input_length=False,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False},
        target_is_mask=False,
        target_is_singlech=False,
        test_num_mics=args.test_num_mics)

    cov = None
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            print('(%d/%d) computing spatial covariance of "%s"' % (idx, total_num, name), flush=True)
            batch = [(name, js[name])]
            # x: Tensor(B, T, C, F)
            x, _ = load_inputs_and_targets(batch)

            # convert numpy.ndarray to torch.tensor (B=1, T, C, F)
            if x[0].dtype.kind == 'c':
                x_real = torch.from_numpy(x[0].real).float()
                x_imag = torch.from_numpy(x[0].imag).float()
                x_new = ComplexTensor(x_real, x_imag).unsqueeze(0)
            elif isinstance(x[0], torch.Tensor):
                # batch of input sequences (B, Tmax, idim)
                x_new = x[0].unsqueeze(0)
            else:
                error = ("x must be numpy.ndarray, torch.Tensor or a dict like "
                         "{{'real': torch.Tensor, 'imag': torch.Tensor}}, "
                         "but got {}".format(type(x[0])))
                raise ValueError(error)

            # x (B, T, C, F) -> Y (B, F, C, T)
            Y = x_new.permute(0, 3, 2, 1)
            # Calculate power: (..., C, T)
            power = Y.real ** 2 + Y.imag ** 2
            # Averaging along the channel axis: (B, F, C, T) -> (B, F, T)
            power = power.mean(dim=-2)
            inverse_power = 1 / torch.clamp(power, min=args.eps)

            # covariance matrix: (B, F, (btaps+1) * C, (btaps+1) * C)
            covariance_matrix = get_covariances(Y, inverse_power, args.bdelay, args.btaps, get_vector=False)
            # normalize to max abs = 1
            covariance_matrix = covariance_matrix / covariance_matrix.abs().max()

            cov = covariance_matrix if cov is None else cov + covariance_matrix

        cov = cov / total_num
        # cov = cov / cov.abs().max()
        torch.save(cov, join(args.output_path, "global_cov.pt"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=str, required=True,
                        help='Filename of train label data (json)')
    parser.add_argument('--preprocess-conf', type=str, default=None, nargs='?',
                        help='The configuration file for the pre-processing')
    parser.add_argument('--output-path', type=str, required=True,
                        help='')
    parser.add_argument('--test-num-mics', type=int, default=-1,
                        help='set number of microphones during testing, same as training by default')
    parser.add_argument('--eps', default=1e-7, type=float,
                        help='Epsilon constant')
    parser.add_argument('--bdelay', default=3, type=int,
                        help='Prediction delay for WPE and WPD')
    parser.add_argument('--btaps', default=5, type=int,
                        help='Number of filter taps')
    args = parser.parse_args()
    compute_global_statistics(args)
