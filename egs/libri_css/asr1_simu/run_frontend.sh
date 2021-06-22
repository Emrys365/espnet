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
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1
n_iter_processes=   # number of workers for data loading

# configuration path
preprocess_config=conf/preprocess.yaml
train_config=frontend/conf/train_multispkr_no_wpe.yaml
decode_config=conf/decode.yaml

# network architecture
num_spkrs=2
# optimizer
lr=

# use IRM as the training target
target_is_mask=
train_mask_only=
mask_loss=
mask_type=
target_is_singlech=
enh_loss_type=
# whether to use oracle IRM for enhancement
test_oracle=

# use VAD-like masks instead of T-F masks, only works when use_padertorch_frontend is True
use_vad_mask=
use_complex_mask=

# only train the WPE module
train_wpe_only=

# set the number of mics during testing (same as traing by default)
test_num_mics=
# set the filter taps (lengths) during testing (same as traing by default)
test_wtaps=
# set the number of input channels for testing (2 by default)
test_nmics=

# use a randomly initialized model for evaluation (not implemented yet)
use_random_model=

use_frontend_for_mix=
# WPD related
use_wpd=
# which version of WPD implementation to be used
#   1: use the AttentionReference to calculate the steering vector for each frequency band
#        (attend along the channel dimension)
#   2: use the AttentionReference to calculate the steering vector, sharing the same values in all frequency bands
#        (attend along the frequency dimension)
#   3: use the original WPD formulas to calculate the steering vector (MaxEigenVector)
#   4: use the modified WPD formulas to calculate the steering vector (MaxEigenVector with Cholesky decomposition)
#   5: use the simplified WPD formulas (MPDR) to get rid of explicit dependence of the steering vector
#   6: use the simplified WPD formulas (MVDR) to get rid of explicit dependence of the steering vector
wpd_opt=1

# use lieaner output layer after beamforming
use_linear_project=
# normalize reference and output speech to max=1
normalization=

#recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
recog_model=model.loss.best

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=SimLibriCSS-short-train-2spk
train_dev=SimLibriCSS-short-dev-2spk
recog_set="SimLibriCSS-short-test-2spk"

if [ -z ${tag} ]; then
    if [ -n "${use_wpd}" ]; then
        expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})${lr:+_lr$lr}${use_linear_project:+_project}${use_vad_mask:+_vad_mask}${enh_loss_type:+_${enh_loss_type}_loss}${test_nmics:+_${test_nmics}ch}_wpd_ver${wpd_opt}
    elif [ -n "${use_frontend_for_mix}" ]; then
        expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})${lr:+_lr$lr}${use_linear_project:+_project}${use_vad_mask:+_vad_mask}${enh_loss_type:+_${enh_loss_type}_loss}${test_nmics:+_${test_nmics}ch}_frontend_for_mix
    else
        expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})${lr:+_lr$lr}${use_linear_project:+_project}${use_vad_mask:+_vad_mask}${enh_loss_type:+_${enh_loss_type}_loss}${test_nmics:+_${test_nmics}ch}
    fi
    #expname=${train_set}_${backend}_b${blayers}_unit${bunits}_proj${bprojs}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
else
    expname=${train_set}_${backend}_${tag}
fi
if [ -n "$train_wpe_only" ]; then
    expname=${expname}_WPEonly
fi
if [ -n "$target_is_mask" ]; then
    if [ -n "$target_is_singlech" ]; then
        expname=${expname}_1chIRM
    else
        expname=${expname}_IRM
    fi
    if [ -n "$train_mask_only" ]; then
        expname=${expname}only_
    fi
else
    if [ -n "$normalization" ]; then
        expname=${expname}_Norm
    fi
fi
if [ -n "$mask_loss" ]; then
    expname=${expname}${mask_loss}
else
    expname=${expname}mse
fi
if [ $ngpu -gt 1 ]; then
    expname=${expname}_${ngpu}gpu
fi
expdir=exp_frontend/${expname} #_ch7_bt8 #_2spk #_test #_bt16
if [ -n "$resume" ]; then
    resume_dir=$(dirname "$resume")
    expdir=${resume_dir%/results}
fi
mkdir -p ${expdir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Data Preparation"
    # generate wav.scp for anechoic data
    for dset in tr cv tt; do
        local/wavdir_to_scp.py \
            ~/work_dir/wyz97/espnet-master/project/wsj0_2mix_spatial/spatialize_wsj0-mix/wsj0-mix/2speakers_anechoic/wav16k/max/${dset}/mix \
            --s1-wav-dir ~/work_dir/wyz97/espnet-master/project/wsj0_2mix_spatial/spatialize_wsj0-mix/wsj0-mix/2speakers_anechoic/wav16k/max/${dset}/s1 \
            --s2-wav-dir ~/work_dir/wyz97/espnet-master/project/wsj0_2mix_spatial/spatialize_wsj0-mix/wsj0-mix/2speakers_anechoic/wav16k/max/${dset}/s2 \
            --output-scp data_frontend/${dset}_spatialized_anechoic_multich/wav.scp
    done

    # generate wav.scp for reverberant data
    for dset in tr cv tt; do
        local/wavdir_to_scp.py \
            ~/work_dir/wyz97/espnet-master/project/wsj0_2mix_spatial/spatialize_wsj0-mix/wsj0-mix/2speakers_reverb/wav16k/max/${dset}/mix \
            --s1-wav-dir ~/work_dir/wyz97/espnet-master/project/wsj0_2mix_spatial/spatialize_wsj0-mix/wsj0-mix/2speakers_reverb/wav16k/max/${dset}/s1 \
            --s2-wav-dir ~/work_dir/wyz97/espnet-master/project/wsj0_2mix_spatial/spatialize_wsj0-mix/wsj0-mix/2speakers_reverb/wav16k/max/${dset}/s2 \
            --output-scp data_frontend/${dset}_spatialized_reverb_multich/wav.scp
    done

    # generate data.json
    local/asr_json_to_ss_json.py \
        --asr-json data/tr_spatialized_reverb_multich_singlespkr/data.json \
        --dataset ${dset} \
        --ss-wav-scp data_frontend/tr_spatialized_reverb_multich/wav.scp data_frontend/tr_spatialized_anechoic_multich/wav.scp \
        > data_frontend/tr_spatialized_reverb_anechoic_multich/data.json

    for dset in cv tt; do
        local/asr_json_to_ss_json.py \
            --asr-json data/${dset}_spatialized_reverb_multich/data.json \
            --dataset ${dset} \
            --ss-wav-scp data_frontend/${dset}_spatialized_reverb_multich/wav.scp \
            > data_frontend/${dset}_spatialized_reverb_anechoic_multich/data.json 
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Filtering out bad samples"

    ### Filter out invalid samples which lead to `loss_ctc=inf` during training.
    # (It takes about one hour.)
    # For consistency, please use the same args as in the training stage.
    tmpdir=$(mktemp -dp ./)
    ${cuda_cmd} --gpu ${ngpu} ${tmpdir}/train.log \
       frontend/filtering_samples.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --seed ${seed} \
        --outdir ${tmpdir}/results \
        --debugmode ${debugmode} \
        --debugdir ${tmpdir} \
        --minibatches ${N} \
        --preprocess-conf ${preprocess_config} \
        --num-spkrs ${num_spkrs} \
        ${use_wpd:+--use-WPD-frontend True} \
        ${use_frontend_for_mix:+--use-wpe-for-mix True} \
        ${wpd_opt:+--wpd-opt $wpd_opt} \
        ${use_linear_project:+--project True} \
        --train-json "data_frontend/${train_set}/data_wpd_maxeig.json" \
        --output-json-path "data_frontend/${train_set}/data_wpd_maxeig2.json"
#        --ctc_type 'builtin'
#    rm -rf $tmpdir
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    if [ -e "${expdir}/train.log" ]; then
        count=1
        while [ -e "${expdir}/train.${count}.log" ]; do
            count=$(( count + 1))
        done
        mv "${expdir}/train.log" "${expdir}/train.${count}.log"
    fi
    if [ -e "${expdir}/results/log" ]; then
        count=1
        while [ -e "${expdir}/results/log.${count}" ]; do
            count=$(( count + 1))
        done
        if [ -e "${expdir}/results/log" ]; then
            mv "${expdir}/results/log" "${expdir}/results/log.${count}"
        fi
        if [ -e "${expdir}/results/loss.png" ]; then
            mkdir -p "${expdir}/results/images${count}"
            mv "${expdir}"/results/*.png "${expdir}/results/images${count}"
        fi
    fi

    set -x
	${cuda_cmd} --qos qd7 --mem 30G --gpu ${ngpu} ${expdir}/train.log \
        frontend/frontend_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --seed ${seed} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json "data/${train_set}/data.json" \
        --valid-json "data/${train_dev}/data.json" \
        --preprocess-conf ${preprocess_config} \
        --num-spkrs ${num_spkrs} \
        --use-padertorch-frontend True \
        ${use_complex_mask:+--use-complex-mask $use_complex_mask} \
        ${n_iter_processes:+--n-iter-processes $n_iter_processes} \
        ${use_wpd:+--use-WPD-frontend True} \
        ${use_frontend_for_mix:+--use-wpe-for-mix True} \
        ${wpd_opt:+--wpd-opt $wpd_opt} \
        ${lr:+--lr $lr} \
        ${use_linear_project:+--project True} \
        ${normalization:+--normalization True} \
        ${target_is_mask:+--target-is-mask True} \
        ${mask_type:+--mask-type $mask_type} \
        ${mask_loss:+--mask-loss $mask_loss} \
        ${target_is_singlech:+--target-is-singlech $target_is_singlech} \
        ${train_mask_only:+--train-mask-only $train_mask_only} \
        ${train_wpe_only:+--train-wpe-only $train_wpe_only} \
        ${test_nmics:+--test-nmics $test_nmics} \
        ${enh_loss_type:+--enh-loss-type $enh_loss_type} \
        ${use_vad_mask:+--use-vad-mask True}

#        --train-json "data_train/${train_set}/data_dereverb.json" \
#        --valid-json "data_train/${train_dev}/data_dereverb.json" \

        #--train-json "data_frontend/${train_set}/data.anechoic_clean.json" \
        #--valid-json "data_frontend/${train_dev}/data.anechoic_clean.json" \
        #--train-json "data_frontend/${train_set}/data.ori_clean.json" \
        #--valid-json "data_frontend/${train_dev}/data.ori_clean.json" \
fi

#if [ -n "$target_is_mask" ] && [ -n "$train_mask_only" ]; then
#    exit 0;
#fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Evaluation"
#    nj=32

    pids=() # initialize pids
    for rtask in ${train_dev} ${train_test}; do
    #for rtask in tt_spatialized_reverb_2ch; do
    #for rtask in tt_spatialized_reverb_librispeech_2ch; do
    (
        decode_dir=eval_SDR_${rtask}_$(basename ${decode_config%.*}) #_no_wpe
        if [ -n "$test_oracle" ]; then
            decode_dir=${decode_dir}_useOracleIRM${test_oracle}
        fi
        if [ -n "$use_random_model" ]; then
            decode_dir=${decode_dir}_use_random_model
        fi
        if [ -n "${test_num_mics}" ] && [ ${test_num_mics} -gt 0 ]; then
            decode_dir=${decode_dir}_${test_num_mics}mics
        fi
        if [ -n "${test_wtaps}" ] && [ ${test_wtaps} -gt 0 ]; then
            decode_dir=${decode_dir}_${test_wtaps}wtaps
        fi
#        decode_dir=${decode_dir}_bfEP14
        feat_recog_dir=data/${rtask}  # ${dumpdir}/${rtask}/delta${do_delta}

        # split data
        #splitjson.py --parts ${nj} ${feat_recog_dir}/data_enh.json

        #### use CPU for decoding
        ngpu=0
#        if [ -f ${expdir}/${decode_dir}/result.log ]; then
#            suffix=$(stat --printf="%y" ${expdir}/${decode_dir}/result.log | cut -d ' ' -f 1 | sed -e 's/-/_/g')
#            count=1
#            if [ -f ${expdir}/${decode_dir}/result.${suffix}.log ]; then
#                while [ -f ${expdir}/${decode_dir}/result.${suffix}_v${count}.log ]; do
#                    let count=count+1
#                done
#                suffix=${suffix}_v${count}
#            fi
#            mv ${expdir}/${decode_dir}/result.log ${expdir}/${decode_dir}/result.${suffix}.log
#        fi

#        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/eval_SDR.JOB.log \
        ${decode_cmd} --mem 20G ${expdir}/${decode_dir}/enhance.log \
            frontend/frontend_test_v2.py \
            --num-spkrs ${num_spkrs} \
            --preprocess-conf ${preprocess_config} \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/data_enh.json \
            --model ${expdir}/results/${recog_model} \
            ${model_wpd:+--model-wpd $model_wpd} \
            ${model_wpe:+--model-wpe $model_wpe} \
            ${model_beamformer:+--model-beamformer $model_beamformer} \
            ${model_beamformer_espnet2:+--model-beamformer-espnet2 $model_beamformer_espnet2} \
            --decode-dir ${expdir}/${decode_dir} \
            ${use_wpd:+--use-WPD-frontend True} \
            ${use_frontend_for_mix:+--use-wpe-for-mix True} \
            ${wpd_opt:+--wpd-opt $wpd_opt} \
            ${use_linear_project:+--project True} \
            ${test_oracle:+--test-oracle $test_oracle} \
            ${target_is_singlech:+--target-is-singlech $target_is_singlech} \
            ${use_random_model:+--use-random-model $use_random_model} \
            ${test_wtaps:+--test-wtaps $test_wtaps} \
            ${test_num_mics:+--test-num-mics $test_num_mics} \
            ${load_from_mdl:+--load-from-mdl $load_from_mdl} \
            ${model_conf:+--model-conf $model_conf} \
            --outdir ${expdir}/results \
            --debugmode ${debugmode} \
            --debugdir ${expdir} \
            --minibatches ${N} \
            --verbose ${verbose} \
            --train-json "data/${train_set}/data_enh.json" \
            --valid-json "data/${train_dev}/data_enh.json" \
            --preprocess-conf ${preprocess_config} \
            --num-spkrs ${num_spkrs}

        wait
#            --recog-json ${feat_recog_dir}/data.anechoic_clean.json \
#            --recog-json ${feat_recog_dir}/data.ori_clean.json \
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
    exit 0;
fi
