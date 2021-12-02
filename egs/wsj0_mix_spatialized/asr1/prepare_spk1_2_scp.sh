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
    "exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_globalcmvn_2ch_5taps_2021_05_14"
    "exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_globalcmvn_2ch_5taps_2021_05_20"
    "exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_globalcmvn_2ch_5taps_vad_mask_2021_05_25"
    "exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_globalcmvn_2ch_5taps_vad_mask_2021_06_03"
)

for subset in cv_spatialized_reverb_multich_16k_max tt_spatialized_reverb_multich_16k_max; do
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
