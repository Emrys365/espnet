#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=1        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
dumpdir=dump   # directory to dump full features
seed=1

# feature configuration
do_delta=false

# configuration path
#decode_config=conf/decode.yaml
decode_config=conf/tuning/decode_pytorch_transformer.yaml

# network architecture
num_spkrs=2

# decoding parameter
use_wordlm=true     # false means to train/use a character LM
lm_weight=1.0
ctc_weight=0.3
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

#train_set=tr_spatialized_reverb_multich_naraWPE_1iter
train_set=train_si284
train_dev=cv_dev93
train_test=test_eval92


dict=data/lang_1char/tr_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
train_set=${train_set}_singlespkr

lmexpdir=exp/train_rnnlm_pytorch_lm_word65000

#expdir=exp_revb_icassp2020/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_wmpdr_atf_preprocess_globalcmvn_5taps_2020_09_15
#recog_model=snapshot.ep.21

######expdir=exp_revb/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_mvdr_wpe_trans_preprocess_globalcmvn_5taps
#ctc_weight=0.7

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    pids=() # initialize pids
    for rtask in ${train_dev} ${train_test}; do
    #for rtask in ${train_dev}; do
    #for rtask in ${train_test}; do
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
        decode_dir=decode${prefix}_$(basename ${decode_config%.*})_${recog_model}_ctcw${ctc_weight}_rnnlm${lm_weight}/${rtask}
        mkdir -p ${expdir}/${decode_dir}

        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        if [ ${lm_weight} == 0 ]; then
            recog_opts=""
        fi
        #if [ -n "${test_nmics}" ]; then
        #    feat_recog_dir=data_8ch/${rtask}  # ${dumpdir}/${rtask}/delta${do_delta}
        #else
            feat_recog_dir=data/${rtask}  # ${dumpdir}/${rtask}/delta${do_delta}
        #fi

        # split data
        #splitjson.py --parts ${nj} ${feat_recog_dir}/data.json
        if [[ $ngpu -eq 0 ]]; then
            #### use CPU for decoding

            #for jobid in 10 11; do
            #(
            #${decode_cmd} --qos qd3 JOB=$jobid ${expdir}/${decode_dir}/log/decode.JOB.log \
            ${decode_cmd} --qos qd3 JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                asr_recog.py \
                --num-spkrs ${num_spkrs} \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model} \
                ${seed:+--seed $seed} \
                ${use_vad_mask:+--use-vad-mask True} \
                ${test_btaps:+--test-btaps $test_btaps} \
                ${test_nmics:+--test-nmics $test_nmics} \
                ${lm_weight:+--lm-weight $lm_weight} \
                ${ctc_weight:+--ctc-weight $ctc_weight} \
                ${recog_opts}

            #wait
            #) &
            #done
        else
            #### use GPU for decoding
            ${decode_cmd} --gpu ${ngpu} --qos qd3 ${expdir}/${decode_dir}/log/decode.log \
                asr_recog.py \
                --num-spkrs ${num_spkrs} \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --recog-json ${feat_recog_dir}/data.json \
                --result-label ${expdir}/${decode_dir}/data.json \
                --model ${expdir}/results/${recog_model} \
                ${seed:+--seed $seed} \
                ${use_vad_mask:+--use-vad-mask True} \
                ${test_btaps:+--test-btaps $test_btaps} \
                ${test_nmics:+--test-nmics $test_nmics} \
                ${lm_weight:+--lm-weight $lm_weight} \
                ${ctc_weight:+--ctc-weight $ctc_weight} \
                ${recog_opts}
        fi

#            --model-conf '/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.3/multi-channel/wsj-mix-spatialized/exp_revb/seed1_tr_spatialized_reverb_multich_naraWPE3_singlespkr2c_pytorch_train_mvdr_wpe_trans_preprocess_init_asr_globalcmvn_5taps_2020_05_09/results/model.json' \
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} --num_spkrs ${num_spkrs} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
    exit 0;
fi
