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
    "exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_globalcmvn_2ch_5taps_2021_05_14"
    "exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_globalcmvn_2ch_5taps_2021_05_20"
    "exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_globalcmvn_2ch_5taps_vad_mask_2021_05_25"
    "exp_revb/seed1_tr_spatialized_anechoic_reverb_16k_max_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_globalcmvn_2ch_5taps_vad_mask_2021_06_03"
)
pids=() # initialize pids
for expdir in "${expdirs[@]}"; do
(
    if [ "$expdir" = "org" ]; then
        for f in cv_spatialized_reverb_multich_16k_max tt_spatialized_reverb_multich_16k_max; do
            echo -e "\ndata/${f}"
            python3 ./cal_srmr.py --ref_channel 0 data/${f}/wav.scp data/${f}/srmr_org_ch0.scp
        done

    else
        for subset in cv_spatialized_reverb_multich_16k_max tt_spatialized_reverb_multich_16k_max; do
            for enhdir in evalSDR_2ch_5btaps_model.acc.best evalSDR_6ch_5btaps_model.acc.best; do
                substr="${expdir}/evaluate_frontend/${enhdir}/${subset}"
                echo -e "\n${substr}"

                ${decode_cmd} --qos qd3 ${substr}/log/srmr1.log \
                    python3 ./cal_srmr.py --ref_channel 0 "${substr}/spk1.scp" "${substr}/srmr1.scp"

                ${decode_cmd} --qos qd3 ${substr}/log/srmr2.log \
                    python3 ./cal_srmr.py --ref_channel 0 "${substr}/spk2.scp" "${substr}/srmr2.scp"
            done
        done
    fi
) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
echo "Finished"
