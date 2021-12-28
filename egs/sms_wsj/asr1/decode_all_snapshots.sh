#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
seed=1

# configuration path
decode_config=conf/tuning/decode_pytorch_transformer.yaml

# network architecture
num_spkrs=2

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lmtag=              # tag for managing LMs

# decoding parameter
lm_weight=0
ctc_weight=0.3

#testing hyperparameters
test_btaps=5 #3
test_nmics=2 #6


expdir= #exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_2021_05_22


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# train_set=train_si284_singlespkr
train_dev=cv_dev93
train_test=test_eval92

dict=data/lang_1char/tr_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt
lmexpdir=exp/train_rnnlm_pytorch_lm_word65000

if [ -z "$expdir" ]; then
    echo "Please specify --expdir"
    exit 1
fi

echo "stage 5: Decoding"
lm_weight=0     # do not use LM
nj=8
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

    if [ ${use_wordlm} = true ]; then
        recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
    else
        recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
    fi
    if [ ${lm_weight} == 0 ]; then
        recog_opts=""
    fi
    feat_recog_dir=data/${rtask}

    splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

    for snapshot_ep in $(ls -tr ${expdir}/results/snapshot.ep.*); do
        snapshot=$(basename "$snapshot_ep")
        decode_dir=decode_all_snapshots_${prefix}_$(basename ${decode_config%.*})_ctcw${ctc_weight}_rnnlm${lm_weight}${lmtag:+_$lmtag}/${rtask}/${snapshot}
        mkdir -p ${expdir}/${decode_dir}

        if [ -e "${expdir}/${decode_dir}/min_perm_result.wrd.json" ]; then
            if grep -e 'Total Scores' -e 'Error Rate' -q "${expdir}/${decode_dir}/min_perm_result.wrd.json"; then
                echo "Skipping ${snapshot} as 'min_perm_result.wrd.json' already exists..."
                continue
            fi
        fi

        ${decode_cmd} --gpu 1 --qos qd3 JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --num-spkrs ${num_spkrs} \
            --config ${decode_config} \
            --ngpu 1 \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${snapshot} \
            ${seed:+--seed $seed} \
            ${test_btaps:+--test-btaps $test_btaps} \
            ${test_nmics:+--test-nmics $test_nmics} \
            ${lm_weight:+--lm-weight $lm_weight} \
            ${ctc_weight:+--ctc-weight $ctc_weight} \
            ${recog_opts}
        wait

        score_sclite.sh --wer true --nlsyms ${nlsyms} --num_spkrs ${num_spkrs} ${expdir}/${decode_dir} ${dict}
        rm ${expdir}/${decode_dir}/data.*.json
    done

) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
echo "Finished"
