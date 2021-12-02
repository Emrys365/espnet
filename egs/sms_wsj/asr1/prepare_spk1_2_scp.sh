#!/bin/bash

set -e

failure() {
  local lineno=$1
  local msg=$2
  echo -e "($0) \033[31mFailed\033[0m at \033[33mline $lineno\033[0m: $msg"
  echo "Exiting..."
}
trap 'failure ${LINENO} "$BASH_COMMAND"' ERR INT


expdirs=(
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_2021_05_22"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_2ch_5taps_2021_05_22"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_vad_mask_2021_05_22"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_2ch_5taps_vad_mask_2021_05_22"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp_frontend/train_si284_pytorch_train_multispkr_mvdr_preprocess_lr0.0001_ci_sdr_loss_2chmse"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_enh_loss_bp_2021_06_21"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_init_frontend_init_asr_uttcmvn_2ch_5taps_enh_loss_bp_2021_06_22"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_2ch_5taps_enh_loss_bp_2021_06_21"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_vad_mask_enh_loss_bp_2021_06_21"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_2ch_5taps_vad_mask_enh_loss_bp_2021_06_21"
)

for subset in cv_dev93 test_eval92; do
    prefix="/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_enh/egs2/sms_wsj/enh1/data/sms_wsj/2speakers/observation/${subset}/"
    for expdir in "${expdirs[@]}"; do
        for folder in "${expdir}/evaluate_frontend"/evalSDR_*; do
            substr="${folder}/${subset}"
            awk -F " " -v enhdir="$substr/enhanced/" '{print $1, enhdir $1 "_0.wav"}' data/${subset}/wav.scp > ${substr}/spk1.scp
            awk -F " " -v enhdir="$substr/enhanced/" '{print $1, enhdir $1 "_1.wav"}' data/${subset}/wav.scp > ${substr}/spk2.scp

            [[ -e "$(head -n 1 "${substr}/spk1.scp" | cut -d' ' -f 2)" ]] || echo "Something wrong with ${substr}/spk1.scp"
            [[ -e "$(head -n 1 "${substr}/spk2.scp" | cut -d' ' -f 2)" ]] || echo "Something wrong with ${substr}/spk2.scp"
        done
    done
done
