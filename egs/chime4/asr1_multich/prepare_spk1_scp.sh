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
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_2ch_5taps_2021_11_10"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_2ch_5taps_vad_mask_2021_11_10"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_tbptt_preprocess_uttcmvn_5taps_2021_11_11"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_tbptt_preprocess_uttcmvn_5taps_vad_mask_2021_11_11"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_2021_11_10"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_vad_mask_2021_11_10"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_tbptt_preprocess_uttcmvn_5taps_2021_11_11"
    "seed1_tr05_multi_isolated_6ch_track_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_tbptt_preprocess_uttcmvn_5taps_vad_mask_2021_11_11"
)

for subset in dt05_simu_isolated_6ch_track et05_simu_isolated_6ch_track; do
   prefix="/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_recipe/egs2/chime4/enh1/dump/\(raw\|raw/org\)/${subset}/data/wav/format.[0-9]\+/data_wav/"
   for expdir in "${expdirs[@]}"; do
       substr="/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/chime4/asr1_multich/exp/${expdir}/evaluate_frontend/evalSDR_5ch_5btaps_model.acc.best_3iter/${subset}"
       sed -e "s#${prefix}#${substr}/enhanced/#g" data/${subset}/wav.scp > ${substr}/spk1.scp
       sed -i -e "s#SIMU\.wav\$#SIMU_0.wav#g" ${substr}/spk1.scp

       [[ -e "$(head -n 1 "${substr}/spk1.scp" | cut -d' ' -f 2)" ]] || exit 1
   done
done

for subset in dt05_real_isolated_6ch_track et05_real_isolated_6ch_track; do
    prefix="/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/espnet_recipe/egs2/chime4/enh1/\(dump_asr\|dump\)/\(raw\|raw/org\)/${subset}/data/wav/format.[0-9]\+/data_wav/"
    for expdir in "${expdirs[@]}"; do
        substr="/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/chime4/asr1_multich/exp/${expdir}/evaluate_frontend/evalSRMR_5ch_5btaps_model.acc.best/${subset}"
        if [ ! -d "${substr}/enhanced" ]; then
            echo "Skipping '$expdir/evaluate_frontend/evalSRMR_5ch_5btaps_model.acc.best/${subset}' because no 'enhanced' directory is found"
            continue
        fi
        sed -e "s#${prefix}#${substr}/enhanced/#g" data/${subset}/wav.scp > ${substr}/spk1.scp
        sed -i -e "s#REAL\.wav\$#REAL_0.wav#g" ${substr}/spk1.scp

        [[ -e "$(head -n 1 "${substr}/spk1.scp" | cut -d' ' -f 2)" ]] || exit 1
    done
done
