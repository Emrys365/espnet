#!/bin/bash

stage=4
stop_stage=5
ngpu=1
train_opt=

# 0 for spatial, reverberant, with WPE module
# 1 for spatial, anechoic data + reverberant data, with separation first in the front-end
# 2 for spatial, reverberant data with nara-WPE preprocessing (5 iterations) + reverberant data, using WPD frontend
task="0"

# which version of WPD implementation to be used
#   1: use the AttentionReference to calculate the steering vector for each frequency band
#        (attend along the channel dimension)
#   2: use the AttentionReference to calculate the steering vector, sharing the same values in all frequency bands
#        (attend along the frequency dimension)
#   3: use the original WPD formulas to calculate the steering vector (MaxEigenVector)
#   4: use the modified WPD formulas to calculate the steering vector (MaxEigenVector with Cholesky decomposition)
#   5: use the simplified WPD formulas (MPDR) to get rid of explicit dependence of the steering vector
#   5.3: use the factorized WPD formulas (WPE + wMPDR)
#   6: use the simplified WPD formulas (MVDR) to get rid of explicit dependence of the steering vector
wpd_opt=
# use a linear projection layer after beamformer
use_linear_project=
normalization=
test_oracle=

use_vad_mask=

# only train the WPE module
train_wpe_only=

# set the number of mics during training
test_nmics=2
# set the number of mics during testing (same as traing by default)
test_num_mics=2
# set the filter taps (lengths) during testing (same as traing by default)
test_wtaps=

# use IRM as the training target
target_is_mask=
train_mask_only=
target_is_singlech=1
mask_loss=
#mask_loss="l1"
#mask_loss="smooth_l1"
#mask_loss="mse"
mask_type=
enh_loss_type="ci_sdr"

if [[ "$task" == "0" ]]; then
    jobname=SSmvdr_cisdr #ss_revb
    wpd_opt=
elif [[ "$task" == "1" ]]; then
    jobname=SSfront
    use_frontend_for_mix=True
    wpd_opt=
elif [[ "$task" == "2" ]]; then
    jobname=SSwpd${wpd_opt}
    use_wpd=True
else
    echo "Error: invalid task id: $task"
    exit 1
fi
if [ -n "$use_linear_project" ]; then
    jobname=${jobname}P
fi
if [ -n "$test_oracle" ]; then
    jobname=${jobname}O
fi
if [ -n "$target_is_mask" ]; then
    jobname=${jobname}M
    if [ -n "$train_mask_only" ]; then
        jobname="${jobname}|"
    fi
    if [ -n "mask_loss" ]; then
        jobname=${jobname}${mask_loss}
    fi
else
    if [ -n "$normalization" ]; then
        jobname=${jobname}N
    fi
fi

if [ -n "${test_num_mics}" ] && [ ${test_num_mics} -gt 0 ]; then
    jobname=${jobname}${test_num_mics}ch
fi
if [ -n "${test_wtaps}" ] && [ ${test_wtaps} -gt 0 ]; then
    jobname=${jobname}${test_wtaps}tap
fi

#elayers_sd=1; elayers_rec=2; eunits=1024; eprojs=$eunits; train_opt="--batchsize 10 --mtleta 0 --mtlalpha 0.2"
#elayers_sd=2; elayers_rec=1; eunits=1024; eprojs=$eunits; train_opt="--mtleta 0.1 --mtlalpha 0.2 --batchsize 10"

#elayers_sd=2; elayers_rec=1; eunits=1024; eprojs=$eunits; dlayers=1; dunits=300; train_opt="--mtleta 0.1 --mtlalpha 0.2 --batchsize 8"

# wpe related
#if [[ "$task" == "2" ]] || [[ "$task" == "4" ]] || [[ "$task" == "6" ]] || [[ "$task" == "7" ]] || [[ "$task" == "8" ]] || [[ "$task" == "9" ]] || [[ "$task" == "b" ]]; then
seed=1

# initial learning rate
lr=0.005
# whether or not to include category info in each minibatch
#with_category=True

#jobname=evalSS
#train_opt="${train_opt} --resume /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_multich/exp_frontend/SimLibriCSS-short-train_pytorch_train_multispkr_no_wpe_preprocess_lr0.7_IRMonly_mse_ch7_bt8/results_old3/snapshot.ep.19"

log_file=log/log.reverb.${jobname}.stage${stage}-${stop_stage}${use_vad_mask:+_vad_mask}${test_nmics:+_${test_nmics}ch}${enh_loss_type:+_${enh_loss_type}_loss}
#log_file=log/log.reverb.WPD5.${jobname}${lr:+_lr$lr}.stage${stage}-${stop_stage}
if [[ $ngpu -gt 1 ]]; then
    log_file=${log_file}_${ngpu}gpu
fi
if [ -n "$test_oracle" ]; then
    log_file=${log_file}_oracle_irm
fi
if [[ "$with_category" == "True" ]]; then
    log_file=${log_file}_with_category
fi
if [ -n "$target_is_mask" ]; then
    log_file=${log_file}_mask
    if [ -n "mask_loss" ]; then
        log_file=${log_file}_${mask_loss}
    fi
else
    if [ -n "$normalization" ]; then
        log_file=${log_file}_Normalized
    fi
fi
if [ -n "${test_num_mics}" ] && [ ${test_num_mics} -gt 0 ]; then
    log_file=${log_file}_${test_num_mics}chInput
fi
if [ -n "${test_wtaps}" ] && [ ${test_wtaps} -gt 0 ]; then
    log_file=${log_file}_${test_wtaps}taps
fi
echo "Log is in $log_file"


#################################################
#                   set jobname                 #
#################################################
run_cmd=run_frontend.sh
echo -e "========================\n        stage: ${stage}\n========================\n"

sbatch_opt="-p cpu --exclude=cqxx-01-00[1-6],gqxx-01-011 --qos qd7"

set -x
sbatch ${sbatch_opt} -J $jobname -o $log_file \
  ${run_cmd} \
    --stage $stage \
    --stop-stage $stop_stage \
    --backend pytorch \
    --ngpu $ngpu \
    --seed $seed \
    ${lr:+--lr $lr} \
    ${use_frontend_for_mix:+--use-frontend-for-mix True} \
    ${use_wpd:+--use-wpd True} \
    ${wpd_opt:+--wpd-opt $wpd_opt} \
    ${use_linear_project:+--use-linear-project True} \
    ${normalization:+--normalization True} \
    ${test_oracle:+--test-oracle True} \
    ${target_is_mask:+--target-is-mask True} \
    ${train_mask_only:+--train-mask-only True} \
    ${mask_loss:+--mask-loss $mask_loss} \
    ${mask_type:+--mask-type $mask_type} \
    ${target_is_singlech:+--target-is-singlech True} \
    ${test_wtaps:+--test-wtaps $test_wtaps} \
    ${test_num_mics:+--test-num-mics $test_num_mics} \
    ${test_nmics:+--test-nmics $test_nmics} \
    ${train_wpe_only:+--train-wpe-only True} \
    ${enh_loss_type:+--enh-loss-type $enh_loss_type} \
    ${use_vad_mask:+--use-vad-mask True} \
    ${train_opt} \
    $@
