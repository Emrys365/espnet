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
pids=() # initialize pids
for expdir in "${expdirs[@]}"; do
(
    if [ "$expdir" = "org" ]; then
        for f in cv_dev93 test_eval92; do
            echo -e "\ndata/${f}"
            python3 ./cal_srmr.py --ref_channel 0 data/${f}/wav.scp data/${f}/srmr_org_ch0.scp
        done

    else
        for subset in cv_dev93 test_eval92; do
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
