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

# Whether to store enhanced outputs
store_output=

# frontend network architecture
use_vad_mask=


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_dev=dt_real_8ch_multich
train_test=et_real_8ch_multich


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Enhancing"

    pids=() # initialize pids
    for rtask in ${train_dev} ${train_test}; do
    (
        prefix=""
        if [ -n "${test_nmics}" ]; then
            prefix=${prefix}_${test_nmics}ch
        fi
        if [ -n "${test_btaps}" ]; then
            prefix=${prefix}_${test_btaps}btaps
        fi
        output_dir=evaluate_frontend/evalSRMR${prefix}_${recog_model}/${rtask}
        #mkdir -p ${expdir}/${output_dir}/enhanced

        #### use CPU for inference
        ${decode_cmd} --qos qd3 ${expdir}/${output_dir}/eval_ss_real.log \
            python3 frontend/eval_raw_srmr.py \
            --mask-type "PSM^2" \
            --data-dir data/${rtask} \
            ${store_output:+--output-dir ${expdir}/${output_dir}/enhanced} \
            --model-path ${expdir}/results/${recog_model} \
            ${test_btaps:+--test-btaps $test_btaps} \
            ${test_nmics:+--test-nmics $test_nmics}

        wait

        #    --output-dir ${expdir}/${output_dir}/enhanced \
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
    exit 0;
fi
