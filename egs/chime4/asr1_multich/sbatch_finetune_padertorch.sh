#!/bin/bash

stage=4
stop_stage=5
ngpu=1
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
#   5.3: use the factorized WPD formulas (WPE + wMPDR)
#   6: use the simplified WPD formulas (MVDR) to get rid of explicit dependence of the steering vector
wpd_opt= #5.3

use_padertorch_frontend=1
use_transformer=1
joint_training=
use_vad_mask=
bf_wpe_tag=
batch_size=6

freeze_frontend=true
freeze_asr=false
use_spec_augment=true

preprocess_config=conf/preprocess.yaml

# number of input channels for training
test_nmics= #5

# data scheduling
multich_epochs=


if [[ "$task" == "0" ]]; then
    jobname=spatial
elif [[ "$task" == "1" ]]; then
    jobname=revbMVDR
elif [[ "$task" == "2" ]]; then
    #jobname=rDWPE2c
    #jobname=factorVAD
    #jobname=factorATF
    #jobname=factor
    jobname=mvdr
    #jobname=mimoSSperm
#    jobname=mimo_notricks
#    jobname=mimo_diag
#    jobname=mimo_db
#    jobname=mimo_maskfl
#    jobname=mimo_solver
    #jobname=mimoVAD
    #jobname=mimoATF
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
    #jobname=wpd_souden
    jobname=wpdsVAD
    #jobname=wpd
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
seed=1

# initial learning rate
lr=
# whether or not to include category info in each minibatch
#with_category=True

# init asr model
if [[ "$task" == "2" ]]; then
#-----------------------------------------------------------------------
    #===== padertorch-frontend, WPE+MVDR_souden =====
    expname=exp/seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_tbptt_preprocess_uttcmvn_5taps_2021_11_11
    #===== padertorch-frontend, WPE+MVDR_atf (2-iter) =====
    #expname=exp/seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_tbptt_preprocess_uttcmvn_5taps_2021_11_11
    #===== padertorch-frontend, WPE+MVDR_souden, VAD-like masks =====
    #expname=exp/seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_tbptt_preprocess_uttcmvn_5taps_vad_mask_2021_11_11
    #===== padertorch-frontend, WPE+MVDR_atf (2-iter), VAD-like masks =====
    #expname=exp/seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_tbptt_preprocess_uttcmvn_5taps_vad_mask_2021_11_11

    init_asr="${expname}/results/model.acc.best"
    init_frontend="${expname}/results/model.acc.best"
    init_from_mdl=

elif [[ "$task" == "b" ]]; then
    init_asr=
    init_frontend=
    init_from_mdl=
else
    init_asr=
    init_frontend=
    init_from_mdl=
fi

#train_opt="${train_opt} --resume /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.3/multi-channel/wsj-mix-spatialized/exp_revb/tr_spatialized_reverb_multich_singlespkr_pytorch_train_multispkr_wyz97_preprocess_arch2/results/snapshot.ep.5"

#log_file=log/log.reverb.finetune.mvdr_wpe.${use_transformer:+_transformer.}stage${stage}-${stop#_stage}
#log_file=log/log.reverb.finetune.mvdr.${use_transformer:+_transformer.}stage${stage}-${stop_stage}
#log_file=log/log.rever.padertorchb.mvdr.${use_transformer:+_transformer.}.no_tricks.stage${stage}-${stop_stage}
#log_file=log/log.rever.padertorchb.mvdr.${use_transformer:+_transformer.}.diagonal_loading.stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.padertorch.mvdr.${use_transformer:+_transformer.}.double_precision.stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.padertorch.mvdr.${use_transformer:+_transformer.}.mask_flooring.stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.padertorch.mvdr.${use_transformer:+_transformer.}.torch_solver.stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.arch1.seed50.${use_transformer:+_transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.wpd_souden${use_vad_mask:+.vad_mask}${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.wpd${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.padertorch.wpe_mwpdr${use_vad_mask:+.vad_mask}.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.padertorch.wpe_mwpdr${use_vad_mask:+.vad_mask}.ss_loss_perm.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.padertorch.wpe_mwpdr_atf${use_vad_mask:+.vad_mask}.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.padertorch.wpe_mwpdr_atf${use_vad_mask:+.vad_mask}.ss_loss_perm.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
log_file=log/log.reverb.finetune.padertorch.wpe_mvdr${use_vad_mask:+.vad_mask}.tBPTT.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}${joint_training:+_joint_training}stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.padertorch.wpe_mvdr${use_vad_mask:+.vad_mask}.ss_loss_perm.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.finetune.padertorch.wpe_mvdr_atf${use_vad_mask:+.vad_mask}.tBPTT.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
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
if [[ "$task" == "0" ]]; then
    run_cmd=run_withsingle3_wyz97.sh
elif [[ "$task" == "1" ]] || [[ "$task" == "2" ]]; then
    if [ -n "$use_transformer" ]; then
        if [ -n "$joint_training" ]; then
            run_cmd=run_withsingle3_reverb_transformer_with_ss_wyz97.sh
            train_opt="${train_opt} --bp-enh-loss True --enh-loss-weight 1.0 --enh-loss-type ci_sdr"
        else
            run_cmd=run_finetune_withsingle3_reverb_transformer_wyz97.sh
        fi
    else
        run_cmd=run_withsingle3_reverb_with_ss_wyz97.sh
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
        #run_cmd=run_reverb_wpd_transformer_new_with_ss.sh
        run_cmd=run_reverb_wpd_transformer_new.sh
    else
        run_cmd=run_withsingle3_reverb_wpd_with_ss.sh
    fi
fi
echo -e "========================\n        stage: ${stage}\n========================\n"

#sbatch_opt="-p cpu --exclude=cqxx-01-00[1-6],cqxx-00-00[1-6],gqxx-01-011"
sbatch_opt="-p cpu --exclude=cqxx-01-00[1-6],gqxx-01-011,gqxx-01-138 --qos qd7"

set -x
sbatch ${sbatch_opt} -J $jobname -o $log_file \
  ${run_cmd} \
    --stage $stage \
    --stop-stage $stop_stage \
    --backend pytorch \
    --ngpu $ngpu \
    --seed $seed \
    ${lr:+--lr $lr} \
    ${wpd_opt:+--wpd-opt $wpd_opt} \
    ${use_spec_augment:+--use-spec-augment $use_spec_augment} \
    ${freeze_frontend:+--freeze-frontend $freeze_frontend} \
    ${freeze_asr:+--freeze-asr $freeze_asr} \
    ${init_asr:+--init-asr $init_asr} \
    ${init_frontend:+--init-frontend $init_frontend} \
    ${init_from_mdl:+--init-from-mdl $init_from_mdl} \
    ${multich_epochs:+--multich-epochs $multich_epochs} \
    ${use_padertorch_frontend:+--use-padertorch-frontend True} \
    ${use_vad_mask:+--use-vad-mask True} \
    ${batch_size:+--batch-size $batch_size} \
    ${bf_wpe_tag:+--bf-wpe-tag $bf_wpe_tag} \
    ${test_nmics:+--test-nmics $test_nmics} \
    ${train_opt} \
    ${preprocess_config:+--preprocess-config $preprocess_config} \
    "$@"
