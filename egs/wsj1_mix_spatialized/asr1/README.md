# End-to-End Far-Field Speech Recognition with Uniﬁed Dereverberation and Beamforming

This recipe provides source code for data simulation and models related to the paper "End-to-End Far-Field Speech Recognition with Uniﬁed Dereverberation and Beamforming".

## Prerequisite Installation
1. Kaldi and ESPnet (v.0.5.3 in this repository)
2. MATLAB (for data simulation)

## Steps
1. This recipe uses spatialized wsj1-2mix as the dataset, and the data simulation scripts and instructions can be found in [data_simu/](https://github.com/Emrys365/espnet/blob/wsj1_mix_spatialized/egs/wsj1_mix_spatialized/asr1/data_simu).

2. After generating the spatialized wsj1-2mix data (16 kHz, `max` version), just run [run.sh](https://github.com/Emrys365/espnet/blob/wsj1_mix_spatialized/egs/wsj1_mix_spatialized/asr1/run.sh) to start data preparation, training and evaluation. You can specify some arguments to control which stages to run, e.g.
```bash
./run.sh --stage 4 --stop-stage 5
```

> Note: You may need to modify the default paths specified in [run.sh](https://github.com/Emrys365/espnet/blob/wsj1_mix_spatialized/egs/wsj1_mix_spatialized/asr1/run.sh) to make it work.

## Note
1. The implementation of frontend modules (both `WPE+MVDR` and `WPD`) can be found in https://github.com/Emrys365/espnet/blob/wsj1_mix_spatialized/espnet/nets/pytorch_backend/frontends/frontend.py and https://github.com/Emrys365/espnet/blob/wsj1_mix_spatialized/espnet/nets/pytorch_backend/frontends/frontend_wpd.py respectively.
