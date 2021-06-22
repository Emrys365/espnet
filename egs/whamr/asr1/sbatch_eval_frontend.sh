#!/bin/bash

stage=5
stop_stage=5

# testing hyper-parameters
test_btaps=5
test_nmics=2


#jobname=iMIMOpd
#jobname=mvdrPD
#jobname=facVAD
#jobname=facATF
#jobname=mimoSSperm
jobname=mimoVADatf
#jobname=mimoATF
#jobname=mimoPD
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


log_file=log/log.eval_ss.reverb.${jobname}${test_btaps:+.${test_btaps}taps}${test_nmics:+.${test_nmics}mics}
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

run_cmd=run_eval_frontend.sh
echo -e "========================\n        stage: ${stage}\n========================\n"

sbatch_opt="-p cpu --exclude=cqxx-01-00[1-6],gqxx-01-011 --qos qd3"

set -x
sbatch ${sbatch_opt} -J $jobname -o $log_file \
  ${run_cmd} \
    --stage $stage \
    --stop-stage $stop_stage \
    ${test_btaps:+--test-btaps $test_btaps} \
    ${test_nmics:+--test-nmics $test_nmics} \
    ${expdir:+--expdir $expdir} \
    ${recog_model:+--recog_model $recog_model}
