#!/bin/bash


#expdir=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_preprocess_uttcmvn_2ch_5taps_2021_05_22
expdir=/mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/espnet-v.0.7.0/egs/sms_wsj/asr1/exp/seed1_train_si284_singlespkr2c_pytorch_train_multispkr_trans_wyz97_padertorch_mvdr_atf_preprocess_uttcmvn_2ch_5taps_vad_mask_2021_05_22

tasks=("cv_dev93" "test_eval92")
num_tasks=${#tasks[@]}
task_i=1
s="{\n"
for rtask in "${tasks[@]}"; do
    wer=()
    sdr=()
    si_sdr=()
    sir=()
    sar=()
    pesq=()
    stoi=()
    srmr=()
    for ep in $(seq 100); do
        decode_file="${expdir}/decode_all_snapshots__2ch_5btaps_decode_pytorch_transformer_ctcw0.3_rnnlm0/${rtask}/snapshot.ep.snapshot.ep.${ep}/min_perm_result.wrd.json"
        if [ -e "$decode_file" ]; then
            wer_=$(grep -Po '(?<=Error Rate:)\s+\d+\.\d+' "$decode_file" | sed -e 's#\s\+##g')
            wer+=("$wer_")
        fi

        eval_dir="${expdir}/evaluate_frontend_all_snapshots/${rtask}/snapshot.ep.${ep}"
        if [ -d "$eval_dir" ]; then
            if [ -e "${eval_dir}/result_sdr.txt" ]; then
                sdr+=("$(sed -e 's#\s##g' "${eval_dir}/result_sdr.txt")")
            fi
            if [ -e "${eval_dir}/result_si_sdr.txt" ]; then
                si_sdr+=("$(sed -e 's#\s##g' "${eval_dir}/result_si_sdr.txt")")
            fi
            if [ -e "${eval_dir}/result_sir.txt" ]; then
                sir+=("$(sed -e 's#\s##g' "${eval_dir}/result_sir.txt")")
            fi
            if [ -e "${eval_dir}/result_sar.txt" ]; then
                sar+=("$(sed -e 's#\s##g' "${eval_dir}/result_sar.txt")")
            fi
            if [ -e "${eval_dir}/result_pesq.txt" ]; then
                pesq+=("$(sed -e 's#\s##g' "${eval_dir}/result_pesq.txt")")
            fi
            if [ -e "${eval_dir}/result_stoi.txt" ]; then
                stoi+=("$(sed -e 's#\s##g' "${eval_dir}/result_stoi.txt")")
            fi
            if [ -e "${eval_dir}/result_srmr.txt" ]; then
                srmr+=("$(sed -e 's#\s##g' "${eval_dir}/result_srmr.txt")")
            fi
        fi
    done

    s+="   \"${rtask}\": {\n"

    counter=0
    s+='      "wer": ['
    num_wers=${#wer[@]}
    for wer_ in "${wer[@]}"; do
        counter=$((counter + 1))
        if [ $counter -eq $num_wers ]; then
            s+="${wer_}],\n"
        else
            s+="${wer_}, "
        fi
    done

    counter=0
    s+='      "sdr": ['
    num_sdrs=${#sdr[@]}
    for sdr_ in "${sdr[@]}"; do
        counter=$((counter + 1))
        if [ $counter -eq $num_sdrs ]; then
            s+="${sdr_}],\n"
        else
            s+="${sdr_}, "
        fi
    done

    counter=0
    s+='      "si_sdr": ['
    num_si_sdrs=${#si_sdr[@]}
    for si_sdr_ in "${si_sdr[@]}"; do
        counter=$((counter + 1))
        if [ $counter -eq $num_si_sdrs ]; then
            s+="${si_sdr_}],\n"
        else
            s+="${si_sdr_}, "
        fi
    done

    counter=0
    s+='      "sir": ['
    num_sirs=${#sir[@]}
    for sir_ in "${sir[@]}"; do
        counter=$((counter + 1))
        if [ $counter -eq $num_sirs ]; then
            s+="${sir_}],\n"
        else
            s+="${sir_}, "
        fi
    done

    counter=0
    s+='      "sar": ['
    num_sars=${#sar[@]}
    for sar_ in "${sar[@]}"; do
        counter=$((counter + 1))
        if [ $counter -eq $num_sars ]; then
            s+="${sar_}],\n"
        else
            s+="${sar_}, "
        fi
    done

    counter=0
    s+='      "pesq": ['
    num_pesqs=${#pesq[@]}
    for pesq_ in "${pesq[@]}"; do
        counter=$((counter + 1))
        if [ $counter -eq $num_pesqs ]; then
            s+="${pesq_}],\n"
        else
            s+="${pesq_}, "
        fi
    done

    counter=0
    s+='      "stoi": ['
    num_stois=${#stoi[@]}
    for stoi_ in "${stoi[@]}"; do
        counter=$((counter + 1))
        if [ $counter -eq $num_stois ]; then
            s+="${stoi_}],\n"
        else
            s+="${stoi_}, "
        fi
    done

    counter=0
    s+='      "srmr": ['
    num_srmrs=${#srmr[@]}
    for srmr_ in "${srmr[@]}"; do
        counter=$((counter + 1))
        if [ $counter -eq $num_srmrs ]; then
            s+="${srmr_}]\n"
        else
            s+="${srmr_}, "
        fi
    done

    if [ $task_i -eq $num_tasks ]; then
        s+="   }\n"
    else
        s+="   },\n"
    fi
    task_i=$((task_i + 1))
done

s+="}"
echo -ne "$s"
