#!/bin/bash

stage=4
stop_stage=5
ngpu=1
train_opt=

#train_opt="${train_opt} --resume /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_2ch_5taps_vad_mask_enh_loss_bp_2021_06_21/results/snapshot.ep.64"


use_transformer=1
batch_size=16

# number of input channels for training
test_nmics=1 #2 #6

# data scheduling
multich_epochs=
jobname=asr_mix_SMS
if [ -n "$use_transformer" ]; then
    jobname=${jobname}T
fi
if [ -n "$multich_epochs" ]; then
    jobname=${jobname}E${multich_epochs}
fi
if [ -n "$test_nmics" ]; then
    jobname=${jobname}${test_nmics}ch
fi

seed=1

# initial learning rate
lr=

init_asr=

log_file=log/log.reverb.asr_mix.1ch.wpe.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
if [[ $ngpu -gt 1 ]]; then
    log_file=${log_file}_${ngpu}gpu
fi
if [[ "$with_category" == "True" ]]; then
    log_file=${log_file}_with_category
fi
if [ -n "$multich_epochs" ]; then
    log_file=${log_file}_fromEpoch${multich_epochs}
fi
echo "Log is in $log_file"


#################################################
#                   set jobname                 #
#################################################
run_cmd=run_1ch.sh
echo -e "========================\n        stage: ${stage}\n========================\n"


set -x
run.pl $log_file \
  ${run_cmd} \
    --stage $stage \
    --stop-stage $stop_stage \
    --backend pytorch \
    --ngpu $ngpu \
    --seed $seed \
    ${lr:+--lr $lr} \
    ${init_asr:+--init-asr $init_asr} \
    ${multich_epochs:+--multich-epochs $multich_epochs} \
    ${batch_size:+--batch-size $batch_size} \
    ${test_nmics:+--test-nmics $test_nmics} \
    ${train_opt} \
    "$@"
