#!/bin/bash

set -e

failure() {
  local lineno=$1
  local msg=$2
  echo -e "($0) \033[31mFailed\033[0m at \033[33mline $lineno\033[0m: $msg"
  echo "Exiting..."
}
trap 'failure ${LINENO} "$BASH_COMMAND"' ERR INT

. ./cmd.sh
. ./path.sh


expdirs=(
    "org"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_5taps_2021_05_21"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_5taps_2021_05_21"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_5taps_vad_mask_2021_05_21"
    "/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/whamr/asr1/exp/seed1_tr_mix_both_anechoic_reverb_max_16k_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_5taps_vad_mask_2021_05_21"
)
pids=() # initialize pids
for expdir in "${expdirs[@]}"; do
(
    if [ "$expdir" = "org" ]; then
        for f in cv_mix_both_reverb_max_16k tt_mix_both_reverb_max_16k; do
            echo -e "\ndata/${f}"
            python3 ./cal_srmr.py --ref_channel 0 data/${f}/wav.scp data/${f}/srmr_org_ch0.scp
        done

    else
        for subset in cv_mix_both_reverb_max_16k tt_mix_both_reverb_max_16k; do
            substr="${expdir}/evaluate_frontend/evalSDR_2ch_5btaps_model.acc.best/${subset}"
            echo -e "\n${substr}"

            ${decode_cmd} --qos qd3 ${substr}/log/srmr1.log \
                python3 ./cal_srmr.py --ref_channel 0 "${substr}/spk1.scp" "${substr}/srmr1.scp"

            ${decode_cmd} --qos qd3 ${substr}/log/srmr2.log \
                python3 ./cal_srmr.py --ref_channel 0 "${substr}/spk2.scp" "${substr}/srmr2.scp"
        done
    fi
) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
echo "Finished"
