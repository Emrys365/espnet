# Steps for generateing spatialized wsj1-2mix data

1. Go to `1.create-speaker-mixtures`:

(1) First please make sure WSJ0's and WSJ1's wv1 sphere files have already been converted to wav files, using the original folder structure, e.g.
```
PATH_TO_WSJ0/11-1.1/wsj0/si_tr_s/01t/01to030v.wv1
==>
YOUR_PATH/wsj0/si_tr_s/01t/01to030v.wav
```

You can run the following command to convert the format and generate some intermediate data for run.sh:
```bash
local/convert2wav.sh ${wsj0_path} ${wsj1_path} YOUR_PATH
```

Then modify `wsj0root` in `1.create-speaker-mixtures/create_wav_2speakers.m` to YOUR_PATH,
and also modify `wsj_full_wav` in [run.sh](https://github.com/Emrys365/espnet/blob/wsj1_mix_spatialized/egs/wsj1_mix_spatialized/asr1/run.sh) to YOUR_PATH.

(2) Finally run the following command to generate (single-channel) wsj1-2mix data:

```bash
matlab -nojvm -nodesktop -nodisplay -nosplash -r create_wav_2speakers
```

> Note: You may need to modify the default paths specified in `1.create-speaker-mixtures/create_wav_2speakers.m`, e.g.
>
> + `wsj0root`
> + `output_dir16k`
> + `output_dir8k`

2. After generating the wsj1-2mix data, go to `2.spatialize`:

(1) First download the source code of RIR generator and compile it:

```bash
wget --continue -O ./RIR-Generator-master/rir_generator.cpp https://raw.githubusercontent.com/ehabets/RIR-Generator/master/rir_generator.cpp
(cd RIR-Generator-master && mex rir_generator.cpp)
```

(2) Then run `launch_spatialize.sh` to generate the spatialized wsj1-2mix data.

> Note: You may need to modify the default arguments specified in `launch_spatialize.sh`, e.g.
>
> + number of parallel processes: `NUM_JOBS=20`
>
> and also the default paths specified in `2.spatialize/spatialize_wsj0_mix.m`, e.g.
>
> + `data_in_root`