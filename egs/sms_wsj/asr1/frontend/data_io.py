from collections import OrderedDict
import logging

import numpy as np
import soundfile

from espnet.transform.transformation import Transformation


class LoadInputsAndTargets(object):
    """Create a mini-batch from a list of dicts

    >>> batch = [('utt1',
    ...           dict(input=[dict(feat='some.ark:123',
    ...                            filetype='mat',
    ...                            name='input1',
    ...                            shape=[100, 80])],
    ...                output=[dict(tokenid='1 2 3 4',
    ...                             name='target1',
    ...                             shape=[4, 31])]]))
    >>> l = LoadInputsAndTargets()
    >>> feat, target = l(batch)

    :param: str mode: Specify the task mode, "asr" or "tts"
    :param: str preprocess_conf: The path of a json file for pre-processing
    :param: bool load_input: If False, not to load the input data
    :param: bool load_output: If False, not to load the output data
    :param: bool sort_in_input_length: Sort the mini-batch in descending order
        of the input length
    :param: bool use_speaker_embedding: Used for tts mode only
    :param: bool use_second_target: Used for tts mode only
    :param: dict preprocess_args: Set some optional arguments for preprocessing
    :param: Optional[dict] preprocess_args: Used for tts mode only
    """

    def __init__(
        self, preprocess_conf=None,
        load_input=True,
        load_input_lengths=False,
        load_output=True,
        sort_in_input_length=True,
        preprocess_args=None,
        keep_all_data_on_mem=False,
        target_is_mask=False,
        target_is_singlech=False,
        test_num_mics=-1,
     ):
        self._loaders = {}
        if preprocess_conf is not None:
            self.preprocessing = Transformation(preprocess_conf)
            logging.warning(
                '[Experimental feature] Some preprocessing will be done '
                'for the mini-batch creation using {}'
                .format(self.preprocessing))
        else:
            # If conf doesn't exist, this function don't touch anything.
            self.preprocessing = None

        self.load_output = load_output
        self.load_input = load_input
        self.load_input_lengths = load_input_lengths
        self.sort_in_input_length = sort_in_input_length
        if preprocess_args is None:
            self.preprocess_args = {}
        else:
            assert isinstance(preprocess_args, dict), type(preprocess_args)
            self.preprocess_args = dict(preprocess_args)

        self.keep_all_data_on_mem = keep_all_data_on_mem
        self.target_is_mask = target_is_mask
        self.target_is_singlech = target_is_singlech
        self.test_num_mics = test_num_mics

    def __call__(self, batch):
        """Function to load inputs and targets from list of dicts

        :param List[Tuple[str, dict]] batch: list of dict which is subset of
            loaded data.json
        :return: list of input token id sequences [(L_1), (L_2), ..., (L_B)]
        :return: list of input feature sequences
            [(T_1, D), (T_2, D), ..., (T_B, D)]
        :rtype: list of float ndarray
        :return: list of target token id sequences [(L_1), (L_2), ..., (L_B)]
        :rtype: list of int ndarray

        """
        x_feats_dict = OrderedDict()  # OrderedDict[str, List[np.ndarray]]
        y_feats_dict = OrderedDict()  # OrderedDict[str, List[np.ndarray]]
        uttid_list = []  # List[str]

        for uttid, info in batch:
            uttid_list.append(uttid)

            if self.load_input:
                # only use the first two channels
                # shape: (T, C)
                nmics = self.test_num_mics if self.test_num_mics > 0 else 2
                x = soundfile.read(info['input'][0]['feat'])[0][:, :nmics]

            if self.load_output:
                # only use the 1st channel as the target label
                # shape: (T,)
                if self.target_is_singlech:
                    y1 = soundfile.read(info['speaker1']['input_feat'])[0]
                    y2 = soundfile.read(info['speaker2']['input_feat'])[0]
                else:
                    if self.target_is_mask:
                        nmics = self.test_num_mics if self.test_num_mics > 0 else 2
                        y1 = soundfile.read(info['speaker1']['input_feat'])[0][:, :nmics]
                        y2 = soundfile.read(info['speaker2']['input_feat'])[0][:, :nmics]
                    else:
                        y1 = soundfile.read(info['speaker1']['input_feat'])[0][:, 0]
                        y2 = soundfile.read(info['speaker2']['input_feat'])[0][:, 0]

                if len(y1) == len(y2) == len(x):
                    x_feats_dict.setdefault('mix_wav', []).append(x)
                    y_feats_dict.setdefault('speaker1_wav', []).append(y1)
                    y_feats_dict.setdefault('speaker2_wav', []).append(y2)
                else:
                    logging.warning('Mismatched sample length: x.shape={}, y1.shape={}, y2.shape={}'.format(x.shape, y1.shape, y2.shape))
            else:
                x_feats_dict.setdefault('mix_wav', []).append(x)


        return_batch, uttid_list = self._create_batch_ss(
            x_feats_dict, y_feats_dict, uttid_list
        )

        if self.preprocessing is not None:
            # Apply pre-processing all input and output features
            for name in return_batch.keys():
                if name == 'mix_wav':
                    # --> (B, T, C, F)
                    return_batch[name] = self.preprocessing(
                        return_batch[name], uttid_list, **self.preprocess_args)
                elif name == 'speaker_wav':
                    # --> Tuple[Tensor(B, T, F), Tensor(B, T, F)]
                    # --> Tuple[Tensor(B, T, C, F), Tensor(B, T, C, F)] (if target_is_mask = True)
                    return_batch[name] = (
                        self.preprocessing(
                        return_batch[name][0], uttid_list, **self.preprocess_args),
                        self.preprocessing(
                        return_batch[name][1], uttid_list, **self.preprocess_args)
                    )

        # Doesn't return the names now.
        return tuple(return_batch.values())

    def _create_batch_ss(self, x_feats_dict, y_feats_dict, uttid_list):
        """Create a OrderedDict for the mini-batch

        :param OrderedDict x_feats_dict:
            e.g. {"input1": [ndarray, ndarray, ...],
                  "input2": [ndarray, ndarray, ...]}
        :param OrderedDict y_feats_dict:
            e.g. {"target1": [ndarray, ndarray, ...],
                  "target2": [ndarray, ndarray, ...]}
        :param: List[str] uttid_list:
            Give uttid_list to sort in the same order as the mini-batch
        :return: batch, uttid_list
        :rtype: Tuple[OrderedDict, List[str]]
        """
        # handle single-input and multi-input (paralell) mode
        xs = list(x_feats_dict.values())

        if self.load_output:
            if len(y_feats_dict) == 1:
                ys = list(y_feats_dict.values())[0]
                assert len(xs[0]) == len(ys), (len(xs[0]), len(ys))

                # get index of non-zero length samples
                nonzero_idx = list(filter(lambda i: len(ys[i]) > 0, range(len(ys))))
            elif len(y_feats_dict) > 1:  # multi-speaker mode
                ys = list(y_feats_dict.values())
                assert len(xs[0]) == len(ys[0]), (len(xs[0]), len(ys[0]))

                # get index of non-zero length samples
                nonzero_idx = list(filter(lambda i: len(ys[0][i]) > 0, range(len(ys[0]))))
                for n in range(1, len(y_feats_dict)):
                    nonzero_idx = filter(lambda i: len(ys[n][i]) > 0, nonzero_idx)
        else:
            # Note(kamo): Be careful not to make nonzero_idx to a generator
            nonzero_idx = list(range(len(xs[0])))

        if self.sort_in_input_length:
            # sort in input lengths based on the first input
            nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[0][i]))
        else:
            nonzero_sorted_idx = list(nonzero_idx)

        if len(nonzero_sorted_idx) != len(xs[0]):
            logging.warning(
                'Target sequences include empty tokenid (batch {} -> {}).'
                .format(len(xs[0]), len(nonzero_sorted_idx)))

        # remove zero-length samples
        xs = [[x[i] for i in nonzero_sorted_idx] for x in xs]
        uttid_list = [uttid_list[i] for i in nonzero_sorted_idx]

        # x_names = list(x_feats_dict.keys())
        x_names = ['mix_wav']
        if self.load_output:
            if len(y_feats_dict) == 1:
                ys = [ys[i] for i in nonzero_sorted_idx]
            elif len(y_feats_dict) > 1:  # multi-speaker asr mode
                # ys = zip(*[[y[i] for i in nonzero_sorted_idx] for y in ys])
                ys = tuple([y[i] for i in nonzero_sorted_idx] for y in ys)

            # y_name= list(y_feats_dict.keys())[0]
            y_name = 'speaker_wav'

            # Keeping x_name and y_name, e.g. input1, for future extension
            return_batch = OrderedDict([*[(x_name, x) for x_name, x in zip(x_names, xs)], (y_name, ys)])
        else:
            return_batch = OrderedDict([(x_name, x) for x_name, x in zip(x_names, xs)])

        if self.load_input_lengths:
            return_batch["wav_lengths"] = [len(wav) for wav in xs[0]]
        return return_batch, uttid_list

