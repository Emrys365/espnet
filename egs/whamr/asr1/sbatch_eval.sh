#!/bin/bash

stage=5
stop_stage=5
ngpu=

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

seed=1

# testing hyper-parameters
lm_weight=1.0
ctc_weight=0.3
test_btaps=5
test_nmics=2

use_vad_mask=

#jobname=iMIMOpd
#jobname=mvdrPD
#jobname=facVAD
#jobname=facATF
#jobname=mimoSSperm
#jobname=mimoVAD
#jobname=mimoATF
jobname=mimoVADatf
#jobname=mimo
#jobname=mimo_notrick
#jobname=mimo_diag
#jobname=mimo_double
#jobname=mimo_maskflr
#jobname=mimo_solver
#jobname=facPD
#jobname=WPDs #souden
#jobname=WPDold
#jobname=MIMOold


log_file=log/log.evaluate.reverb.${jobname}${use_vad_mask:+.vad_mask}.seed${seed}${lm_weight:+.lm${lm_weight}}${ctc_weight:+.ctc${ctc_weight}}${test_btaps:+.${test_btaps}taps}${test_nmics:+.${test_nmics}mics}${ngpu:+_${ngpu}gpu}
echo "Log is in $log_file"

jobname=${jobname}${test_btaps:+${test_btaps}t}${test_nmics:+${test_nmics}ch}


#############################################
#          set experiment directory         #
#############################################
model_opt=3

if [[ $model_opt -eq 0 ]]; then

# padertorch-frontend, WPE+MVDR_souden
expdir=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_5taps_2021_05_21
recog_model=model.acc.best

elif [[ $model_opt -eq 1 ]]; then

# padertorch-frontend, WPE+MVDR_atf (2-iter)
expdir=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_5taps_2021_05_21
recog_model=model.acc.best

elif [[ $model_opt -eq 2 ]]; then

# padertorch-frontend, WPE+MVDR_souden, VAD-like masks
expdir=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_5taps_vad_mask_2021_05_21
recog_model=model.acc.best

elif [[ $model_opt -eq 3 ]]; then

# padertorch-frontend, WPE+MVDR_atf (2-iter), VAD-like masks
expdir=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_5taps_vad_mask_2021_05_21
recog_model=model.acc.best

elif [[ $model_opt -eq 4 ]]; then

# padertorch-frontend, WPE+MVDR_souden, complex T-F masks
expdir=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_complex_preprocess_uttcmvn_5taps_2021_05_28
recog_model=model.acc.best

else
    echo "Invalid model_opt: $model_opt"
    exit 1;
fi

run_cmd=run_eval_reverb_transformer.sh
echo -e "========================\n        stage: ${stage}\n========================\n"

#sbatch_opt="-p cpu --exclude=cqxx-01-00[1-6],cqxx-00-00[1-6],gqxx-01-011"
sbatch_opt="-p cpu --exclude=cqxx-01-00[1-6],gqxx-01-011 --qos qd3"

set -x
sbatch ${sbatch_opt} -J $jobname -o $log_file \
  ${run_cmd} \
    --stage $stage \
    --stop-stage $stop_stage \
    --backend pytorch \
    --seed $seed \
    ${ngpu:+--ngpu $ngpu} \
    ${wpd_opt:+--wpd-opt $wpd_opt} \
    ${use_vad_mask:+--use-vad-mask True} \
    ${test_btaps:+--test-btaps $test_btaps} \
    ${test_nmics:+--test-nmics $test_nmics} \
    ${expdir:+--expdir $expdir} \
    ${recog_model:+--recog_model $recog_model} \
    ${lm_weight:+--lm-weight $lm_weight} \
    ${ctc_weight:+--ctc-weight $ctc_weight}
