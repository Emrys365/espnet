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

max() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -ge "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=
n_iter_processes=   # number of workers for data loading

# feature configuration
do_delta=false

# use padertorch-like frontend (estimating all masks in one network)
use_padertorch_frontend=
# use VAD-like masks instead of T-F masks, only works when use_padertorch_frontend is True
use_vad_mask=
num_nodes=1
lr=

# config files
#preprocess_config=conf/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
preprocess_config=conf/preprocess.yaml
train_config=conf/train_multispkr_padertorch.yaml
#train_config=conf/train_multispkr.yaml

#train_config=conf/train_multispkr_1ch.yaml
#train_config=conf/train_multispkr_no_wpe.yaml
decode_config=conf/decode.yaml
lm_config=conf/lm.yaml

# WPD related
wpd_opt=
use_WPD_frontend=

# multi-speaker asr related
num_spkrs=2         # number of speakers
use_spa=false       # speaker parallel attention

# Initialization
init_frontend=
init_asr=
init_from_mdl=

# Testing hyperparameters
test_btaps=
test_nmics=

# rnnlm related
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
#datadir=/export/a15/vpanayotov/data
datadir=$(pwd)

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

dict=data/lang_char/train_960_unigram5000_units.txt
bpemodel=data/lang_char/train_960_unigram5000
# dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
# bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"

# LibriSpeech corpus direcory
corpus_dir=/mnt/lustre/sjtu/shared/data/asr/rawdata/LibriSpeech

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev
recog_set="test"

#train_set=SimLibriCSS-short-train
#train_dev=SimLibriCSS-short-dev
#recog_set="SimLibriCSS-short-test"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    echo "stage 0: Data preparation"
    ### This part is for LibriCSS training data
    ### Before next step, suppose the training set has been generated (please refer to jsalt2020-asrdiar/jsalt2020_simulate for more details).
    # Prepare transcripts for training data
    python3 local/get_transcript.py ${corpus_dir}/train-clean-100 ${corpus_dir}/train-clean-360 > data/trans_train_clean.txt
    python3 local/get_transcript.py ${corpus_dir}/dev-clean > data/trans_dev_clean.txt
    python3 local/get_transcript.py ${corpus_dir}/test-clean > data/trans_test_clean.txt

    # Get gender information
    ##echo -e "# 2484 speakers\nLibrispeech_spk2gender = {" > dict_spk2gender.py
    echo -e "{" > dict_spk2gender.json
    grep -Po '^\d+\s+\|\s+[FM]' ~/data/asr/rawdata/LibriSpeech/doc/SPEAKERS.TXT | awk -v ORS=' '  -F' *| *' '{print "\"" $1 "\": \"" $3 "\"," }' | sed -e 's/\("[[:digit:]]\{1,\}": "[FM]",\) \("[[:digit:]]\{1,\}": "[FM]",\) \("[[:digit:]]\{1,\}": "[FM]",\) /   \1 \2 \3\n/g' >> dict_spk2gender.json
    echo -e "}" >> dict_spk2gender.json
    
    # Generate Kaldi style data files (wav.scp, utt2spk, text) for training
    for setname in ${train_set} ${train_dev} ${recog_set}; do
        echo "Preparing data for ${setname}..."
#        python3 local/prepare_train_data.py --mixlog /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/jsalt2020_simulate/JelinekWorkshop2020/data/SimLibriUttmix-${setname}/mixlog.json --trans data/trans_${setname}_clean.txt --tgtpath data_train/SimLibriUttmix-${setname}
#        python3 local/prepare_train_data_mtg.py --mixlog /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/jsalt2020_simulate/JelinekWorkshop2020/data/SimLibriCSS-short-${setname}/mixlog.json --trans data/trans_${setname}_clean.txt --tgtpath data_train/SimLibriCSS-short-${setname}

        python3 local/prepare_raw_data_json_mtg.py \
            --mixlog /mnt/lustre/sjtu/users/wyz97/work_dir/wyz97/jsalt2020/jsalt2020_simulate/JelinekWorkshop2020/data/SimLibriCSS-short-${setname}/mixlog.json \
            --tgtpath data_train/SimLibriCSS-short-${setname} \
            --trans data/trans_${setname}_clean.txt \
            --spm_model ${bpemodel}.model \
            --dict ${dict} \
            --max_num_spkrs 2 \
            --preprocess_conf ${preprocess_config}
            
    done

    for setname in ${train_set} ${train_dev} ${recog_set}; do
        # Sort files by utt_id
        for x in wav.scp wav_spk1.scp wav_spk2.scp utt2spk text text_spk1 text_spk2 text_spk3; do
            if [ -f data_train/SimLibriCSS-short-${setname}/${x} ]; then
                mv data_train/SimLibriCSS-short-${setname}/${x} data_train/SimLibriCSS-short-${setname}/.${x}
                cat data_train/SimLibriCSS-short-${setname}/.${x} | sort > data_train/SimLibriCSS-short-${setname}/${x}
                rm data_train/SimLibriCSS-short-${setname}/.${x}
            fi
        done
        utils/utt2spk_to_spk2utt.pl data_train/SimLibriCSS-short-${setname}/utt2spk > data_train/SimLibriCSS-short-${setname}/spk2utt
        # Validate data directory
        if [ -f data_train/SimLibriCSS-short-${setname}/text ]; then
            utils/validate_data_dir.sh --no-feats data_train/SimLibriCSS-short-${setname}
        else
            utils/validate_data_dir.sh --no-feats --no-text data_train/SimLibriCSS-short-${setname}
        fi
    done

    for setname in ${train_set} ${train_dev} ${recog_set}; do
        curdir=$PWD
        cd ./data_train/SimLibriCSS-short-${setname}/
        mkdir -p {spk1,spk2}
        [ -f ./spk1/utt2spk ] || ln -s ../utt2spk ./spk1/utt2spk
        [ -f ./spk1/spk2utt ] || ln -s ../spk2utt ./spk1/spk2utt
        [ -f ./spk2/utt2spk ] || ln -s ../utt2spk ./spk2/utt2spk
        [ -f ./spk2/spk2utt ] || ln -s ../spk2utt ./spk2/spk2utt
        [ -f ./spk1/text ] || ln -s ../text_spk1 ./spk1/text
        [ -f ./spk2/text ] || ln -s ../text_spk2 ./spk2/text
        #sed -e 's/\.wav$/_0.wav/' ./wav.scp > ./spk1/wav.scp
        #sed -e 's/\.wav$/_1.wav/' ./wav.scp > ./spk2/wav.scp
        #sed -e 's/\.wav$/_2.wav/' ./wav.scp > ./noise.scp
        cd $curdir
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Dump wav files into a HDF5 file"
    for setname in ${train_set} ${train_dev} ${recog_set}; do
        dump_pcm.sh --nj ${nj} --cmd "${train_cmd} --mem 20G" --filetype "sound.hdf5" --format flac data_train/SimLibriCSS-short-${setname}
    done

#    for setname in ${train_set} ${train_dev} ${recog_set}; do
#        for subdir in spk1 spk2; do
#            dump_pcm.sh --nj ${nj} --cmd "${train_cmd} --mem 20G" --filetype "sound.hdf5" --format flac data_train/SimLibriCSS-short-${setname}/${subdir}
#        done
#    done
    echo "Done"
    exit 0;
fi

#train_set="${train_set}_multich"
#train_dev="${train_dev}_multich"
# Rename recog_set: e.g. dev -> dev_multich
#recog_set="$(for setname in ${recog_set}; do echo -n "${setname}_multich "; done)"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
#    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
#    cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
#    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
#    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    #wc -l ${dict}

    # make json labels using customized data2json.sh
    # <ignore_id> is used as the transcript for the 2nd "speaker" in single-speaker samples
    #  - id of <ignore_id> is -1
    for setname in ${train_set} ${train_dev} ${recog_set}; do
        local/data2json.sh --cmd "${train_cmd}" --nj ${nj} --num-spkrs 2 \
            --category "multichannel" \
            --preprocess-conf ${preprocess_config} --filetype sound.hdf5 \
            --feat data_train/SimLibriCSS-short-${setname}/feats.scp --bpecode ${bpemodel}.model \
            --out data_train/SimLibriCSS-short-${setname}/data_${bpemode}${nbpe}.json data_train/SimLibriCSS-short-${setname} ${dict}
    done

    for setname in ${train_set} ${train_dev} ${recog_set}; do
        for subdir in spk1 spk2; do
             data2json.sh --cmd "${train_cmd}" --nj ${nj} \
                --preprocess-conf ${preprocess_config} --filetype sound.hdf5 \
                --feat data_train/SimLibriCSS-short-${setname}/${subdir}/feats.scp --bpecode ${bpemodel}.model \
                --out data_train/SimLibriCSS-short-${setname}/${subdir}/data_${bpemode}${nbpe}.json data_train/SimLibriCSS-short-${setname}/${subdir} ${dict}
        done
    done

    for setname in ${train_set} ${train_dev} ${recog_set}; do
        mkdir -p data_train/SimLibriCSS-short-${setname}/mix/
        local/merge_datajsons_ts.py \
            data_train/SimLibriCSS-short-${setname}/data_${bpemode}${nbpe}.json \
            data_train/SimLibriCSS-short-${setname}/spk1/data_${bpemode}${nbpe}.json \
            data_train/SimLibriCSS-short-${setname}/spk2/data_${bpemode}${nbpe}.json \
            --dataset SimLibriCSS-short-${setname} \
            > data_train/SimLibriCSS-short-${setname}/mix/data_${bpemode}${nbpe}.json
    done
fi

# It takes about one day. If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
# lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpname=train_rnnlm_pytorch_lm_transformer_cosine_batchsize32_lr1e-4_layer16_unigram5000_ngpu4
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_${bpemode}${nbpe}
    mkdir -p ${lmdatadir}
    # use external data
    if [ ! -e data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm_train/
    fi
    cut -f 2- -d" " data/${train_set}/text | gzip -c > data/local/lm_train/${train_set}_text.gz
    # combine external text and transcriptions and shuffle them with seed 777
    zcat data/local/lm_train/librispeech-lm-norm.txt.gz data/local/lm_train/${train_set}_text.gz |\
        spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
        > ${lmdatadir}/valid.txt
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
fi

#train_set=SimLibriCSS-short-train-2spk
train_set=SimLibriCSS-short-train-2spk_singlespkr
train_dev=SimLibriCSS-short-dev-2spk
recog_set="SimLibriCSS-short-test-2spk"

if [ -n "$test_nmics" ]; then
    chs="_${test_nmics}ch"
else
    chs=
fi
if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})${chs}${use_vad_mask:+_vad_mask}${init_frontend:+_init_frontend}${init_asr:+_init_asr}${init_from_mdl:+_init_from_mdl}${lr:+_lr$lr}
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ $ngpu -gt 1 ]; then
        expname=${expname}_${ngpu}gpu
    fi
    if [ $num_nodes -gt 1 ]; then
        expname=${expname}_${num_nodes}node
    fi
else
    expname=${train_set}_${backend}${chs}${use_vad_mask:+_vad_mask}${init_frontend:+_init_frontend}${init_asr:+_init_asr}${init_from_mdl:+_init_from_mdl}_${tag}${lr:+_lr$lr}
fi
${use_spa} && spa=true

if [ -z "$resume" ]; then
    expdir=exp/${expname}
else
    resume_dir=$(dirname "$resume")
    expdir=${resume_dir%/results}
fi

#expdir=exp/${expname}_initASRandFrontend_2ch_double_precision_bs4 #_resumeEP3 #_bs4
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    train_opts=""

    if [ -e "${expdir}/train.log" ]; then
        count=1
        while [ -e "${expdir}/train.${count}.log" ]; do
            count=$(( count + 1))
        done
        mv "${expdir}/train.log" "${expdir}/train.${count}.log"
    fi
    if [ -e "${expdir}/results/log" ]; then
        count=1
        while [ -e "${expdir}/results/log.${count}" ]; do
            count=$(( count + 1))
        done
        if [ -e "${expdir}/results/log" ]; then
            mv "${expdir}/results/log" "${expdir}/results/log.${count}"
        fi
        if [ -e "${expdir}/results/acc.png" ]; then
            mkdir -p "${expdir}/results/images${count}"
            mv "${expdir}"/results/*.png "${expdir}/results/images${count}"
        fi
    fi

    jobname="$(basename ${expdir})"
    set -x
    #${cuda_cmd} --gpu ${ngpu} --num-arrays ${num_nodes} --export ALL --mem 100G ${expdir}/train.log \

    python3 -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${expdir}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${expdir}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        asr_train_ddp.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --num-nodes ${num_nodes} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json data/${train_set}/data2.json \
        --valid-json data/${train_dev}/data2.json \
        --num-spkrs ${num_spkrs} \
        --load-wav-ref False \
        ${lr:+--lr $lr} \
        ${n_iter_processes:+--n-iter-processes $n_iter_processes} \
        ${init_frontend:+--init-frontend $init_frontend} \
        ${init_asr:+--init-asr $init_asr} \
        ${init_from_mdl:+--init-from-mdl $init_from_mdl} \
        ${test_nmics:+--test-nmics $test_nmics} \
        ${wpd_opt:+--wpd-opt $wpd_opt} \
        ${use_WPD_frontend:+--use-WPD-frontend True} \
        ${use_padertorch_frontend:+--use-padertorch-frontend True} \
        ${use_vad_mask:+--use-vad-mask True} \
        ${spa:+--spa}
fi

#        --save-interval-iters 1 \

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
#        if ${use_lm_valbest_average}; then
#            lang_model=rnnlm.val${lm_n_average}.avg.best
#            opt="--log ${expdir}/results/log"
#        else
#            lang_model=rnnlm.last${lm_n_average}.avg.best
#            opt="--log"
#        fi
#        average_checkpoints.py \
#            ${opt} \
#            --backend ${backend} \
#            --snapshots ${lmexpdir}/snapshot.ep.* \
#            --out ${lmexpdir}/${lang_model} \
#            --num ${lm_n_average}
#    fi
    nj=32
    #test_nmics=7

    pids=() # initialize pids
    for rtask in ${train_dev} ${recog_set}; do
    (
        prefix=""
        if [ -n "${test_nmics}" ]; then
            prefix=${prefix}_${test_nmics}ch
        fi
        if [ -n "${test_btaps}" ]; then
            prefix=${prefix}_${test_btaps}btaps
        fi
#        lmtag=MaskPostProc
        decode_dir=decode${prefix}_${rtask}_${recog_model}_$(basename ${decode_config%.*})${lmtag:+_$lmtag}
        feat_recog_dir=data/${rtask}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data2.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} --mem 30G JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --num-spkrs ${num_spkrs} \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data2.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --api v2 \
            ${test_btaps:+--test-btaps $test_btaps} \
            ${test_nmics:+--test-nmics $test_nmics}
            # --rnnlm ${lmexpdir}/${lang_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --num_spkrs 2  --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
