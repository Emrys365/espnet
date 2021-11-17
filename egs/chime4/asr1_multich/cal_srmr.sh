#!/bin/bash

set -e

failure() {
  local lineno=$1
  local msg=$2
  echo -e "($0) \033[31mFailed\033[0m at \033[33mline $lineno\033[0m: $msg"
  echo "Exiting..."
}
trap 'failure ${LINENO} "$BASH_COMMAND"' ERR INT

. ./path.sh


expdirs=(
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_2ch_5taps_2021_11_10"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_2ch_5taps_vad_mask_2021_11_10"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_tbptt_preprocess_uttcmvn_5taps_2021_11_11"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_tbptt_preprocess_uttcmvn_5taps_vad_mask_2021_11_11"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_2021_11_10"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_vad_mask_2021_11_10"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_tbptt_preprocess_uttcmvn_5taps_2021_11_11"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_tbptt_preprocess_uttcmvn_5taps_vad_mask_2021_11_11"
)
#for subset in dt05_simu_isolated_6ch_track dt05_real_isolated_6ch_track et05_simu_isolated_6ch_track et05_real_isolated_6ch_track; do
for subset in dt05_simu_isolated_6ch_track et05_simu_isolated_6ch_track; do
    for expdir in "${expdirs[@]}"; do
        substr="/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/chime4/asr1_multich/exp/${expdir}/evaluate_frontend/evalSDR_5ch_5btaps_model.acc.best_3iter/${subset}"
        echo -e "\n${substr}"
        python3 ./cal_srmr.py "${substr}/spk1.scp" "${substr}/srmr1.scp"
    done
done
