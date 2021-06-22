#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=1        # start from 0 if you need to start from data preparation
stop_stage=100

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
#recog_model=snapshot.ep.13

#testing hyperparameters
test_btaps= #3
test_nmics= #6
expdir=
recog_model=


# frontend network architecture
use_vad_mask=


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=SimLibriCSS-short-train-2spk_singlespkr
train_dev=SimLibriCSS-short-dev-2spk
recog_set="SimLibriCSS-short-test-2spk"


#expdir=exp_revb_icassp2020/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_wmpdr_atf_preprocess_globalcmvn_5taps_2020_09_15
#recog_model=snapshot.ep.21

######expdir=exp_revb/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_mvdr_wpe_trans_preprocess_globalcmvn_5taps
#ctc_weight=0.7

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Enhancing"
    nj=32

    pids=() # initialize pids
#    for rtask in ${train_dev} ${train_test}; do
    for rtask in ${train_test}; do
    #for rtask in tt_spatialized_reverb_2ch; do
    #for rtask in tt_spatialized_reverb_librispeech_2ch; do
    (
        prefix=""
        if [ -n "${test_nmics}" ]; then
            prefix=${prefix}_${test_nmics}ch
        fi
        if [ -n "${test_btaps}" ]; then
            prefix=${prefix}_${test_btaps}btaps
        fi
        output_dir=evaluate_frontend/evalSDR${prefix}_${recog_model}/${rtask}
        mkdir -p ${expdir}/${output_dir}/enhanced

        #### use CPU for inference
        ${decode_cmd} ${expdir}/${output_dir}/eval_ss.log \
            python3 frontend/eval_raw.py \
            --data-dir data_ss_eval/${rtask} \
            --model-path ${expdir}/results/${recog_model} \
            --output-dir ${expdir}/${output_dir}/enhanced \
            ${test_btaps:+--test-btaps $test_btaps} \
            ${test_nmics:+--test-nmics $test_nmics}
        wait

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
    exit 0;
fi
