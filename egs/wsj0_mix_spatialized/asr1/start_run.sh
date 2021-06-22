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
use_vad_mask=1
bf_wpe_tag=
batch_size=8

# number of input channels for training
test_nmics=2 #6

# data scheduling
multich_epochs=

# Time-Frequency Masking
#train_opt="${train_opt} --resume /mnt/xlancefs/home/wyz97/workspace/espnet_jsalt2020/egs/wsj0_mix_spatialized/asr1/exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_globalcmvn_2ch_5taps_2021_05_14/results/snapshot.ep.100"
#train_opt="${train_opt} --resume /mnt/xlancefs/home/wyz97/workspace/espnet_jsalt2020/egs/wsj0_mix_spatialized/asr1/exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_globalcmvn_2ch_5taps_2021_05_20/results/snapshot.ep.82"

# VAD-like Masking
#train_opt="${train_opt} --resume /mnt/xlancefs/home/wyz97/workspace/espnet_jsalt2020/egs/wsj0_mix_spatialized/asr1/exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_globalcmvn_2ch_5taps_vad_mask_2021_05_25/results/snapshot.ep.91"
train_opt="${train_opt} --resume /mnt/xlancefs/home/wyz97/workspace/espnet_jsalt2020/egs/wsj0_mix_spatialized/asr1/exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_globalcmvn_2ch_5taps_vad_mask_2021_06_03/results/snapshot.ep.71"

if [[ "$task" == "0" ]]; then
    jobname=spatial
elif [[ "$task" == "1" ]]; then
    jobname=revbMVDRvad
elif [[ "$task" == "2" ]]; then
    #jobname=rDWPE2c
    #jobname=factorVAD
    #jobname=factorATF
    #jobname=factor
    jobname=mvdrVADatf
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
###    init_asr=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/wsj1_mix_spatialized/asr1/exp_revb_icassp2020/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_preprocess_init_frontend_init_asr_globalcmvn_5taps_2020_09_08/results/model.acc.best
###    init_frontend=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/wsj1_mix_spatialized/asr1/exp_revb_icassp2020/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_preprocess_init_frontend_init_asr_globalcmvn_5taps_2020_09_08/results/model.acc.best
   # init_asr='/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.2/multi-channel/wsj-mix-spatialized/exp/tr_spatialized_anechoic_multich_singlespkr_pytorch_b3_unit512_proj512_vggblstmp_e3_subsample1_2_2_1_1_unit1024_proj1024_d1_unit300_location_aconvc10_aconvf100_mtlalpha0.2_adadelta_sampprob0.0_bs8_mli800_mlo150_lsmunigram0.05_globalcmvn_repeat/results/model.acc.best'
#    init_asr='/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.3/multi-channel/wsj-mix-spatialized/exp_revb/seed1_train_si284_pytorch_train_multispkr800_trans_wyz97_preprocess_wpd_ver5_globalcmvn_5taps_2020_05_07/results/snapshot.ep.50'
#    init_frontend='/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.3/multi-channel/wsj-mix-spatialized/exp_revb/seed1_train_si284_pytorch_train_multispkr800_trans_wyz97_preprocess_wpd_ver5_globalcmvn_5taps_2020_05_07/results/snapshot.ep.50'
    #init_asr=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/wsj/asr1/exp/train_si284_pytorch_train_no_preprocess/results/model.last10.avg.best

#-----------------------------------------------------------------------
#    init_asr=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/wsj1_mix_spatialized/asr1/exp_revb_icassp2020_rebuttal/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_notricks_preprocess_globalcmvn_5taps_2021_01_14/results/snapshot.ep.9
#    init_frontend=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/wsj1_mix_spatialized/asr1/exp_revb_icassp2020_rebuttal/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_notricks_preprocess_globalcmvn_5taps_2021_01_14/results/snapshot.ep.9
#-----------------------------------------------------------------------
#    init_asr=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/wsj1_mix_spatialized/asr1/exp_revb_icassp2020_rebuttal/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_double_precision_preprocess_globalcmvn_5taps_2021_01_13/results/snapshot.ep.9
#    init_frontend=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/wsj1_mix_spatialized/asr1/exp_revb_icassp2020_rebuttal/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_double_precision_preprocess_globalcmvn_5taps_2021_01_13/results/snapshot.ep.9
#-----------------------------------------------------------------------
#    init_asr=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/wsj1_mix_spatialized/asr1/exp_revb_icassp2020_rebuttal/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_mask_flooring_preprocess_globalcmvn_5taps_2021_01_13/results/snapshot.ep.9
#    init_frontend=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/wsj1_mix_spatialized/asr1/exp_revb_icassp2020_rebuttal/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_mask_flooring_preprocess_globalcmvn_5taps_2021_01_13/results/snapshot.ep.9
#-----------------------------------------------------------------------
#    init_asr=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/wsj1_mix_spatialized/asr1/exp_revb_icassp2020_rebuttal/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_torch_solver_preprocess_globalcmvn_5taps_2021_01_13/results/snapshot.ep.9
#    init_frontend=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/wsj1_mix_spatialized/asr1/exp_revb_icassp2020_rebuttal/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_torch_solver_preprocess_globalcmvn_5taps_2021_01_13/results/snapshot.ep.9
#-----------------------------------------------------------------------
    init_asr=
    init_frontend=

elif [[ "$task" == "b" ]]; then
    init_frontend=
    init_from_mdl=
else
    init_asr=
    init_frontend=
    init_from_mdl=
fi

#train_opt="${train_opt} --resume /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet-v0.5.3/multi-channel/wsj-mix-spatialized/exp_revb/tr_spatialized_reverb_multich_singlespkr_pytorch_train_multispkr_wyz97_preprocess_arch2/results/snapshot.ep.5"

#log_file=log/log.reverb.mvdr_wpe.${use_transformer:+_transformer.}stage${stage}-${stop#_stage}
#log_file=log/log.reverb.mvdr.${use_transformer:+_transformer.}stage${stage}-${stop_stage}
#log_file=log/log.rever.padertorchb.mvdr.${use_transformer:+_transformer.}.no_tricks.stage${stage}-${stop_stage}
# log_file=log/log.rever.padertorchb.mvdr.${use_transformer:+_transformer.}.diagonal_loading.stage${stage}-${stop_stage}
#log_file=log/log.reverb.padertorch.mvdr.${use_transformer:+_transformer.}.double_precision.stage${stage}-${stop_stage}
#log_file=log/log.reverb.padertorch.mvdr.${use_transformer:+_transformer.}.mask_flooring.stage${stage}-${stop_stage}
#log_file=log/log.reverb.padertorch.mvdr.${use_transformer:+_transformer.}.torch_solver.stage${stage}-${stop_stage}
#log_file=log/log.reverb.arch1.seed50.${use_transformer:+_transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.wpd_souden${use_vad_mask:+.vad_mask}${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.wpd${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.padertorch.wpe_mwpdr${use_vad_mask:+.vad_mask}.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.padertorch.wpe_mwpdr${use_vad_mask:+.vad_mask}.ss_loss_perm.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.padertorch.wpe_mwpdr_atf${use_vad_mask:+.vad_mask}.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.padertorch.wpe_mwpdr_atf${use_vad_mask:+.vad_mask}.ss_loss_perm.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
log_file=log/log.reverb.padertorch.wpe_mvdr${use_vad_mask:+.vad_mask}.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.padertorch.wpe_mvdr${use_vad_mask:+.vad_mask}.ss_loss_perm.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
#log_file=log/log.reverb.padertorch.wpe_mvdr_atf${use_vad_mask:+.vad_mask}.${lr:+_lr$lr}_2c.seed${seed}.${use_transformer:+transformer.}stage${stage}-${stop_stage}
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
        #run_cmd=run_withsingle3_reverb_transformer_with_ss_wyz97.sh
        #train_opt="${train_opt} --bp-enh-loss True"
        run_cmd=run_withsingle3_reverb_transformer_wyz97.sh
        run_cmd=run.sh
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


set -x
run.pl $log_file \
  ${run_cmd} \
    --stage $stage \
    --stop-stage $stop_stage \
    --backend pytorch \
    --ngpu $ngpu \
    --seed $seed \
    ${lr:+--lr $lr} \
    ${wpd_opt:+--wpd-opt $wpd_opt} \
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
    $@
