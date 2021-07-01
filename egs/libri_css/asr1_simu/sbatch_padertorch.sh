#!/bin/bash

stage=4
stop_stage=5
ngpu=4
train_opt=

# 0 for spatial, anechoic
# 1 for spatial, reverberant
# 2 for spatial, reverberant, with WPE module
# 3 for spatial, reverberant data with nara-WPE preprocessing (5 iterations)
# 4 for spatial, reverberant data with nara-WPE preprocessing (5 iterations) + reverberant data
# 5 for spatial, reverberant data with nara-WPE preprocessing (5 iterations) + reverberant data
# 6 for spatial, anechoic data + reverberant data, with ASR model init
# 7 for spatial, anechoic data + reverberant data, with separation first in the front-end
# 8 for spatial, anechoic data + reverberant data, with beamforming before WPE
# 9 for spatial, deverb_reverberant data + reverberant data, with separation first in the front-end
# a for spatial, reverberant data with nara-WPE preprocessing (1 iteration) + reverberant data
# b for spatial, reverberant data with nara-WPE preprocessing (5 iterations) + reverberant data, using WPD frontend
task="2"

# which version of WPD implementation to be used
#   1: use the AttentionReference to calculate the steering vector for each frequency band
#        (attend along the channel dimension)
#   2: use the AttentionReference to calculate the steering vector, sharing the same values in all frequency bands
#        (attend along the frequency dimension)
#   3: use the original WPD formulas to calculate the steering vector (MaxEigenVector)
#   4: use the modified WPD formulas to calculate the steering vector (MaxEigenVector with Cholesky decomposition)
#   5: use the simplified WPD formulas (MPDR) to get rid of explicit dependence of the steering vector
#   6: use the simplified WPD formulas (MVDR) to get rid of explicit dependence of the steering vector
wpd_opt=
use_WPD_frontend=
train_opt=

# number of input channels for training
test_nmics=2 #7

use_transformer=1
use_vad_mask=
num_nodes=2

# data scheduling
multich_epochs=

# Time-Frequency mask
#train_opt="${train_opt} --resume /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_simu/exp/SimLibriCSS-short-train-2spk_singlespkr_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_2ch_4gpu/results/snapshot.ep.12"
#train_opt="${train_opt} --resume /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_simu/exp/SimLibriCSS-short-train-2spk_singlespkr_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_2ch_4gpu/results/snapshot.ep.12"
#train_opt="${train_opt} --resume /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_simu/exp/SimLibriCSS-short-train-2spk_singlespkr_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_adam_init_preprocess_2ch_init_asr_lr0.001_4gpu/results/snapshot.ep.4"
#train_opt="${train_opt} --resume /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_simu/exp/SimLibriCSS-short-train-2spk_singlespkr_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_adam_init_preprocess_2ch_init_asr_lr0.001_4gpu/results/snapshot.ep.2"

# VAD-like mask
#train_opt="${train_opt} --resume /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_simu/exp/SimLibriCSS-short-train-2spk_singlespkr_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_2ch_vad_mask_4gpu/results/snapshot.ep.6"
#train_opt="${train_opt} --resume /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_simu/exp/SimLibriCSS-short-train-2spk_singlespkr_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_2ch_vad_mask_4gpu/results/snapshot.ep.11"


if [[ "$task" == "0" ]]; then
    jobname=spatial
elif [[ "$task" == "1" ]]; then
    jobname=revbMVDR
elif [[ "$task" == "2" ]]; then
    jobname=libriMVDRadamATFinit
    #jobname=rD2c
elif [[ "$task" == "3" ]]; then
    jobname=r_nara
elif [[ "$task" == "4" ]]; then
    jobname=r_na2
elif [[ "$task" == "5" ]]; then
    jobname=r_na3
elif [[ "$task" == "6" ]]; then
    jobname=r_init
elif [[ "$task" == "7" ]]; then
    jobname=r_front
elif [[ "$task" == "8" ]]; then
    jobname=r_BF1st
elif [[ "$task" == "9" ]]; then
    jobname=r_2front
elif [[ "$task" == "a" ]]; then
    jobname=r_1iter
elif [[ "$task" == "b" ]]; then
    jobname=wpd${wpd_opt}_2c
else
    echo "Error: invalid task id: $task"
    exit 1
fi
if [ -n "$use_transformer" ]; then
    jobname=${jobname}T
fi
if [ -n "$multich_epochs" ]; then
    jobname=${jobname}E${multich_epochs}
fi
if [ -n "$test_nmics" ]; then
    jobname=${jobname}${test_nmics}ch
fi


#elayers_sd=1; elayers_rec=2; eunits=1024; eprojs=$eunits; train_opt="--batchsize 10 --mtleta 0 --mtlalpha 0.2"
#elayers_sd=2; elayers_rec=1; eunits=1024; eprojs=$eunits; train_opt="--mtleta 0.1 --mtlalpha 0.2 --batchsize 10"

#elayers_sd=2; elayers_rec=1; eunits=1024; eprojs=$eunits; dlayers=1; dunits=300; train_opt="--mtleta 0.1 --mtlalpha 0.2 --batchsize 8"

# wpe related
#if [[ "$task" == "2" ]] || [[ "$task" == "4" ]] || [[ "$task" == "6" ]] || [[ "$task" == "7" ]] || [[ "$task" == "8" ]] || [[ "$task" == "9" ]] || [[ "$task" == "b" ]]; then
seed=2

# initial learning rate
lr=0.001
# whether or not to include category info in each minibatch
#with_category=True

# init asr model
if [[ "$task" == "2" ]]; then
#    init_asr=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/librispeech/asr1/exp/train_960_pytorch_train_specaug_init_4workers_withCTC/results/model.acc.best

    init_asr=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/librispeech/asr1/exp/SimLibriUtt-train_pytorch_train_specaug_init_4workers_withCTC/results/model.acc.best
#    init_frontend=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_multich/dump_tmp/combined_ss_model.th

    #init_asr=
    init_frontend=
    init_from_mdl=
    #init_frontend=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.3/multi-channel/wsj-mix-spatialized/exp_revb/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_mvdr_wpe_trans_preprocess_globalcmvn_5taps/results/model.acc.best
    #init_asr='librispeech.transformer.v1/exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug/results/model.val5.avg.best'
elif [[ "$task" == "b" ]]; then

    init_asr=
    init_frontend=
    #init_frontend='/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.3/multi-channel/wsj-mix-spatialized/exp_frontend/tr_spatialized_reverb_anechoic_multich_pytorch_train_multispkr_wyz97_preprocess_lr0.7_wpd_ver5_1chIRMl1_1chClean/results/model.loss.best'
    #init_frontend='/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.3/multi-channel/wsj-mix-spatialized/exp_frontend/tr_spatialized_reverb_anechoic_multich_pytorch_train_multispkr512_wyz97_preprocess_lr0.7_wpd_ver5_1chIRMl1_1chClean/results/model.loss.best'
    #init_asr='/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.3/multi-channel/wsj-mix-spatialized/exp_revb/seed1_train_si284_pytorch_train_multispkr800_trans_wyz97_preprocess_wpd_ver5_globalcmvn_5taps_2020_05_07/results/snapshot.ep.50'
#    init_asr=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/librispeech/asr1/exp/SimLibriUtt-train_pytorch_train_specaug_init_4workers_withCTC/results/model.acc.best
#    init_frontend=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_multich/dump_tmp/combined_ss_model.th
    init_from_mdl='/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/libri_css/asr1_multich/libricss_models/MIMO_modular/combined_MIMO_model.th'
else
    init_asr=
    init_frontend=
    init_from_mdl=
fi

#train_opt="${train_opt} --resume /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.3/multi-channel/wsj-mix-spatialized/exp_revb/tr_spatialized_reverb_multich_singlespkr_pytorch_train_multispkr_wyz97_preprocess_arch2/results/snapshot.ep.5"

#log_file=log/log.reverb.mvdr_wpe.${use_transformer:+_transformer.}stage${stage}-${stop_stage}_init
log_file=log/log.reverb.arch1${use_vad_mask:+.vad_mask}.pdtorch.seed${seed}.${use_transformer:+_transformer.}${jobname}.stage${stage}-${stop_stage}${init_frontend:+_init_frontend}${init_asr:+_init_asr}${init_from_mdl:+_init_from_mdl}
#log_file=log/log.reverb.wpd${wpd_opt}${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.1ch.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
if [[ $ngpu -gt 1 ]]; then
    log_file=${log_file}_${ngpu}gpu
fi
if [[ $num_nodes -gt 1 ]]; then
    log_file=${log_file}_${num_nodes}node
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
if [[ "$task" == "0" ]]; then
    run_cmd=run_withsingle3_wyz97.sh
elif [[ "$task" == "1" ]] || [[ "$task" == "2" ]]; then
    if [ -n "$use_transformer" ]; then
        run_cmd=run_withsingle3_reverb_transformer_wyz97.sh
    else
        run_cmd=run_withsingle3_reverb_wyz97.sh
    fi
elif [[ "$task" == "3" ]]; then
    run_cmd=run_withsingle3_reverb_naraWPE.sh
elif [[ "$task" == "4" ]]; then
    run_cmd=run_withsingle3_reverb_naraWPE2.sh
elif [[ "$task" == "5" ]]; then
    run_cmd=run_withsingle3_reverb_naraWPE3.sh
elif [[ "$task" == "6" ]]; then
    run_cmd=run_withsingle3_reverb_init.sh
elif [[ "$task" == "7" ]]; then
    run_cmd=run_withsingle3_reverb_wpe_for_mix.sh
elif [[ "$task" == "8" ]]; then
    run_cmd=run_withsingle3_reverb_beamforming_first.sh
elif [[ "$task" == "9" ]]; then
    run_cmd=run_withsingle3_reverb_wpe_for_mix_dereverb.sh
elif [[ "$task" == "a" ]]; then
    run_cmd=run_withsingle3_reverb_naraWPE_1iter.sh
elif [[ "$task" == "b" ]]; then
    if [ -n "$use_transformer" ]; then
        run_cmd=run_withsingle3_reverb_wpd_transformer.sh
    else
        run_cmd=run_withsingle3_reverb_wpd.sh
    fi
fi
echo -e "========================\n        stage: ${stage}\n========================\n"
if [[ $num_nodes -gt 1 ]]; then
    run_cmd=run_train_ddp.sh
    train_opt="${train_opt} --num-nodes $num_nodes"
else
    run_cmd=run_train.sh
fi

#sbatch_opt="-p cpu --exclude=cqxx-01-00[1-6],cqxx-00-00[1-6],gqxx-01-011"
sbatch_opt="-p cpu --exclude=cqxx-01-00[1-6],gqxx-01-011,gqxx-01-072 --qos qd7"

set -x
sbatch ${sbatch_opt} -J $jobname -o $log_file \
  ${run_cmd} \
    --stage $stage \
    --stop-stage $stop_stage \
    --backend pytorch \
    --ngpu $ngpu \
    --seed $seed \
    --use-padertorch-frontend True \
    ${lr:+--lr $lr} \
    ${wpd_opt:+--wpd-opt $wpd_opt} \
    ${init_asr:+--init-asr $init_asr} \
    ${init_frontend:+--init-frontend $init_frontend} \
    ${init_from_mdl:+--init-from-mdl $init_from_mdl} \
    ${multich_epochs:+--multich-epochs $multich_epochs} \
    ${use_WPD_frontend:+--use-WPD-frontend True} \
    ${use_vad_mask:+--use-vad-mask True} \
    ${test_nmics:+--test-nmics $test_nmics} \
    ${train_opt} \
    $@
