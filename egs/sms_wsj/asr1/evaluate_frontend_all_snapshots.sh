#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

#testing hyperparameters
test_btaps=5 #3
test_nmics=2 #6
expdir= #exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_2021_05_22
num_spkrs=2


. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# train_set=train_si284
train_dev=cv_dev93
train_test=test_eval92

if [ -z "$expdir" ]; then
    echo "Please specify --expdir"
    exit 1
fi

echo "stage 5: Enhancing"
nj=4
pids=() # initialize pids
for rtask in ${train_dev} ${train_test}; do
(
    prefix=""
    if [ -n "${test_nmics}" ]; then
        prefix=${prefix}_${test_nmics}ch
    fi
    if [ -n "${test_btaps}" ]; then
        prefix=${prefix}_${test_btaps}btaps
    fi
    feat_recog_dir=data/${rtask}

    if [ ${nj} -gt 1 ]; then
        _logdir="${feat_recog_dir}/log"
        mkdir -p "${_logdir}"
        # Split the key file
        key_file=${feat_recog_dir}/wav.scp
        split_scps=""
        _nj=$(min "${nj}" "$(<${key_file} wc -l)")
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/wav.${n}.scp"
        done
        utils/split_scp.pl "${key_file}" ${split_scps}

        for spk in $(seq "${num_spkrs}"); do
            key_file=${feat_recog_dir}/spk${spk}.scp
            split_scps=""
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/spk${spk}.${n}.scp"
            done
            utils/split_scp.pl "${key_file}" ${split_scps}
        done
    else
        _nj=1
    fi
    # splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

    for snapshot_ep in $(ls -tr ${expdir}/results/snapshot.ep.*); do
        snapshot=$(basename "$snapshot_ep")
        output_dir=evaluate_frontend_all_snapshots/${rtask}/${snapshot}
        mkdir -p ${expdir}/${output_dir}

        if [ -e "${expdir}/${output_dir}/result_sdr.txt" ]; then
            echo "Skipping ${snapshot} as 'result_sdr.txt' already exists..."
            continue
        fi

        if [ ${_nj} -eq 1 ]; then
            ${decode_cmd} --gpu 1 --qos qd3 ${expdir}/${output_dir}/eval_ss.log \
                python3 frontend/eval_raw_v2.py \
                --use-oracle-mask False \
                --mask-type "PSM^2" \
                --data-dir "${feat_recog_dir}" \
                --model-path ${expdir}/results/${snapshot} \
                ${test_btaps:+--test-btaps $test_btaps} \
                ${test_nmics:+--test-nmics $test_nmics} \
                --verbose False \
                --wav-scp-suffix .scp \
                --write-scps True \
                --write-scp-dir ${expdir}/${output_dir} \
                --write-scp-suffix .scp
        else
            ${decode_cmd} --gpu 1 --qos qd3 JOB=1:${_nj} ${expdir}/${output_dir}/eval_ss.JOB.log \
                python3 frontend/eval_raw_v2.py \
                --use-oracle-mask False \
                --mask-type "PSM^2" \
                --data-dir "${_logdir}" \
                --model-path ${expdir}/results/${snapshot} \
                ${test_btaps:+--test-btaps $test_btaps} \
                ${test_nmics:+--test-nmics $test_nmics} \
                --verbose False \
                --wav-scp-suffix .JOB.scp \
                --write-scps True \
                --write-scp-dir ${expdir}/${output_dir} \
                --write-scp-suffix .JOB.scp

            for metric in si_sdr sdr sir sar pesq stoi srmr; do
                for i in $(seq "${_nj}"); do
                    cat "${expdir}/${output_dir}/${metric}.${i}.scp"
                done | LC_ALL=C sort -k1 > "${expdir}/${output_dir}/${metric}.scp"
                rm "${expdir}/${output_dir}"/${metric}.*.scp
            done
        fi
        wait
            # --output-dir ${expdir}/${output_dir}/enhanced \

        for metric in si_sdr sdr sir sar pesq stoi srmr; do
            awk 'BEGIN{sum=0}
                {n=0;score=0;for (i=2; i<=NF; i+=1){n+=1;score+=$i}; sum+=score/n}
                END{printf ("%.5f\n",sum/NR)}' ${expdir}/${output_dir}/${metric}.scp \
            > "${expdir}/${output_dir}/result_${metric}.txt"
        done
    done

) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
echo "Finished"
