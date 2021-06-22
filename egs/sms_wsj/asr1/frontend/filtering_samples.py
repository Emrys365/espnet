#!/usr/bin/env python3

# Copyright 2019 Shanghai Jiao Tong University (Wangyou Zhang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import json
import logging
import math
import sys

from chainer import training
from chainer.training.updater import StandardUpdater
import torch
from torch.nn.parallel import data_parallel

from batchfy import make_batchset
from data_io import LoadInputsAndTargets
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator

from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from frontend_train import get_parser
from separation import CustomConverter
from ss_model import E2E


class CustomUpdater(StandardUpdater):
    """Custom Updater for Pytorch.

    Args:
        model (torch.nn.Module): The model to update.
        train_iter (chainer.dataset.Iterator): The training iterator.
        optimizer (torch.optim.optimizer): The training optimizer.

        device (torch.device): The device to use.
        ngpu (int): The number of gpus to use.
        use_apex (bool): The flag to use Apex in backprop.

    """

    def __init__(self, model, train_iter, converter,
                 optimizer, device, ngpu):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.converter = converter
        self.device = device
        self.ngpu = ngpu
        self.iteration = 0
        self.invalid_samples = []

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main update routine of the CustomUpdater."""
        train_iter = self.get_iterator('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.next()
        x = self.converter(batch, self.device)

        # Compute the loss at this time step and accumulate it
        self.model.eval()

        # no gradient and backward
        with torch.no_grad():
            # Compute the loss at this time step and accumulate it
            if self.ngpu == 0:
                loss = self.model(*x).mean()
            else:
                # apex does not support torch.nn.DataParallel
                loss = data_parallel(self.model, x, range(self.ngpu)).mean()
        loss_data = float(loss)

        if loss_data >= CTC_LOSS_THRESHOLD or math.isnan(loss_data):
            self.invalid_samples.append(self.iteration)
            logging.warning('Invalid sample detected!')

    def update(self):
        self.update_core()
        self.iteration += 1


def filtering_train_json(train_json, sample_ids):
    """Filtering out the invalid samples from the original train_json.

    Args:
        train_json (dict): Dictionary of training data.
        sample_ids (list): List of ids of samples to be filtered out.

    Returns:
        new_train_json (dict): Filtered dictionary of training data.
    """
    new_train_json = train_json.copy()
    for sample in sample_ids:
        new_train_json.pop(sample)
        print("'{}' removed".format(sample))
    return new_train_json


def pseudo_train(args):
    """Pretend to train with the given args to check if the training samples are valid.

    Bad samples will be detected and reported if they lead to ``loss=nan``.

   The training data will be sorted in the order of input length (long to short) and not shuffled.

    Args:
        args (namespace): The program arguments.
    """
    # overridden some arguments to save time and memory cost
    args.batch_size = 1
    args.epochs = 1
    args.lsm_type = ''
    args.rnnlm = None
    args.ngpu = 1
    args.sortagrad = 0

    set_deterministic_pytorch(args)

    # get input and output dimension info
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    utt0 = next(iter(train_json))
    idim = int(train_json[utt0]['shape'][-1])
    odim = idim
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify model architecture
    model = E2E(args, idim)
    subsampling_factor = model.subsample[0]

    reporter = model.reporter

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(subsampling_factor=subsampling_factor, dtype=dtype)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']

    # make minibatch list (batch_size = 1)
    # from long to short
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=False)

    load_tr = LoadInputsAndTargets(
        load_output=True, sort_in_input_length=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    train_iter = ToggleableShufflingSerialIterator(
        TransformDataset(train, load_tr),
        batch_size=1, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(model, train_iter, converter, optimizer, device, args.ngpu)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Run the training
    trainer.run()

    # ids of the invalid samples
    sample_ids = [train[i][0][0] for i in updater.invalid_samples]
    return filtering_train_json(train_json, sample_ids)


if __name__ == '__main__':
    cmd_args = sys.argv[1:]
    parser = get_parser(required=False)
    parser.add_argument('--output-json-path', type=str, required=True,
                        help='Output path of the filtered json file')
    args, _ = parser.parse_known_args(cmd_args)

    if args.model_module is None:
        model_module = "espnet.nets." + args.backend + "_backend.e2e_asr:E2E"
    else:
        model_module = args.model_module
    model_class = dynamic_import(model_module)
    model_class.add_arguments(parser)

    args = parser.parse_args(cmd_args)
    args.model_module = model_module
    if 'chainer_backend' in args.model_module:
        args.backend = 'chainer'
    if 'pytorch_backend' in args.model_module:
        args.backend = 'pytorch'

    # load dictionary
    if args.dict is not None:
        with open(args.dict, 'rb') as f:
            dictionary = f.readlines()
        char_list = [entry.decode('utf-8').split(' ')[0]
                     for entry in dictionary]
        char_list.insert(0, '<blank>')
        char_list.append('<eos>')
        args.char_list = char_list
    else:
        args.char_list = None

    # get filtered train_json without invalid samples
    new_train_json = pseudo_train(args)
    jsonstring = json.dumps({'utts': new_train_json}, indent=4, ensure_ascii=False, sort_keys=True)
    with open(args.output_json_path, 'w') as f:
        f.write(jsonstring)
