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

# general configuration
nj=32
enh_exp=

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

test_sets="dev_multich"


if [ -z "$enh_exp" ]; then
    echo "please specify --enh_exp"
    exit 0
fi

_cmd=${decode_cmd}
pids=() # initialize pids
for dset in ${test_sets}; do
(
    _inf_dir="${enh_exp}/enhanced_${dset}"
    _dir="${enh_exp}/enhanced_${dset}/metrics"
    _logdir="${_dir}/logdir"
    mkdir -p "${_logdir}"

    # split data
    ref_file=data/${dset}/spk1.scp
    split_scps=""
    _nj=$(min "${nj}" "$(<${ref_file} wc -l)")
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/ref.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${ref_file}" ${split_scps}

    enh_file=${_inf_dir}/spk1.scp
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/enh.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${enh_file}" ${split_scps}

    ${_cmd} JOB=1:"${_nj}" "${_logdir}"/evaluate_metric.JOB.log \
        python evaluate_metric.py \
            --ref_scp "${_logdir}"/ref.JOB.scp \
            --enh_scp "${_logdir}"/enh.JOB.scp \
            --outdir "${_logdir}"/output.JOB \
            --out_suffix .scp

    for protocol in stoi wer metric; do
        for i in $(seq "${_nj}"); do
            cat "${_logdir}/output.${i}/${protocol}.scp"
        done | LC_ALL=C sort -k1 > "${_dir}/${protocol}.scp"
    done

    for protocol in stoi wer metric; do
        # shellcheck disable=SC2046
        awk 'BEGIN{sum=0}
            {n=0;score=0;for (i=2; i<=NF; i+=1){n+=1;score+=$i}; sum+=score/n}
            END{print sum/NR}' "${_dir}"/"${protocol}.scp" > "${_dir}/result_${protocol,,}.txt"
    done

) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
echo "Finished"
