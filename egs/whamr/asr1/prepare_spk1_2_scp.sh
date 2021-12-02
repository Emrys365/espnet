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
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_5taps_2021_05_21"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_5taps_2021_05_21"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_5taps_vad_mask_2021_05_21"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_5taps_vad_mask_2021_05_21"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_complex_preprocess_uttcmvn_5taps_2021_05_28"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_5taps_enh_loss_bp_2021_07_02"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_5taps_enh_loss_bp_2021_07_02"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_5taps_vad_mask_enh_loss_bp_2021_07_02"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_5taps_vad_mask_enh_loss_bp_2021_07_02"
)

for subset in cv tt; do
    prefix="/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_enh/egs2/whamr/enh1/data/whamr/2speakers/wav16k/max/${subset}/mix_both_reverb/"
    for expdir in "${expdirs[@]}"; do
        for folder in "${expdir}/evaluate_frontend"/evalSDR_*; do
            if [ ! -d "$folder" ]; then
                echo "Skipping '${expdir}' due to lack of 'evaluate_frontend'"
                continue
            fi
            substr="${folder}/${subset}_mix_both_reverb_max_16k"
            if [ ! -d "$substr" ]; then
                echo "Skipping '${substr}' (missing)"
                continue
            fi
            awk -F " " -v enhdir="$substr/enhanced/" '{print $1, enhdir $1 "_0.wav"}' data/${subset}_mix_both_reverb_max_16k/wav.scp > ${substr}/spk1.scp
            awk -F " " -v enhdir="$substr/enhanced/" '{print $1, enhdir $1 "_1.wav"}' data/${subset}_mix_both_reverb_max_16k/wav.scp > ${substr}/spk2.scp

            [[ -e "$(head -n 1 "${substr}/spk1.scp" | cut -d' ' -f 2)" ]] || echo "Something wrong with ${substr}/spk1.scp"
            [[ -e "$(head -n 1 "${substr}/spk2.scp" | cut -d' ' -f 2)" ]] || echo "Something wrong with ${substr}/spk2.scp"
        done
    done
done
