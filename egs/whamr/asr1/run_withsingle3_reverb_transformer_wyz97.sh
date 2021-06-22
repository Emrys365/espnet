#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=1        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false
fbank_fs=16000

# configuration path
preprocess_config=conf/preprocess.yaml  # use conf/specaug.yaml for data augmentation
train_config=conf/tuning/train_multispkr_trans_wyz97_padertorch_mvdr.yaml
decode_config=conf/tuning/decode_pytorch_transformer.yaml

# network architecture
num_spkrs=2
batch_size=

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_layers=1         # 2 for character LMs
lm_units=1000       # 650 for character LMs
lm_opt=sgd          # adam for character LMs
lm_sortagrad=0 # Feed samples from shortest to longest ; -1: enabled for all epochs, 0: disabled, other: enabled for 'other' epochs
lm_batchsize=300    # 1024 for character LMs
lm_epochs=20        # number of epochs
lm_patience=3
lm_maxlen=40        # 150 for character LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
lm_weight=1.0
beam_size=30
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
#recog_model=snapshot.ep.13

#initialization
init_asr=
init_frontend=
init_from_mdl=

#testing hyperparameters
test_btaps= #3
test_nmics= #6

# data
wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/mnt/lustre/sjtu/users/xkc09/asr/wsj_kaldi/wsj1
wsj_full_wav=$PWD/data/wsj0/wsj0_wav
wsj_2mix_wav=$PWD/data/wsj0_mix/2speakers
wsj_2mix_scripts=$PWD/data/wsj0_mix/scripts

# frontend network architecture
use_padertorch_frontend=
use_vad_mask=
bf_wpe_tag=

# data scheduling
multich_epochs=

# cmvn
stats_file= #fbank/tr_spatialized_all/cmvn.ark
apply_uttmvn=true

# exp tag
tag="" # tag for managing experiments.
cmvn_tag="uttcmvn"

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#train_set=tr_spatialized_reverb_multich_naraWPE_1iter
train_set=tr_mix_both_reverb_max_16k
train_aux_set=wsj_train_si284
train_dev=cv_mix_both_reverb_max_16k
train_test=tt_mix_both_reverb_max_16k
recog_set="tt_mix_both_reverb_max_16k"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh
fi

feat_tr_dir=data/${train_set}; #${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=data/${train_dev}; #${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    #echo "stage 1: Feature Generation"
    #fbankdir=fbank
    ## Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    #for x in train_si284 test_dev93 test_eval92; do
    #    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
    #        data/${x} exp/make_fbank/${x} ${fbankdir}
    #done

    ## compute global CMVN
    #compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    ## dump features for training
    #if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    #utils/create_split_dir.pl \
    #    /export/b{10,11,12,13}/${USER}/espnet-data/egs/wsj/asr1/dump/${train_set}/delta${do_delta}/storage \
    #    ${feat_tr_dir}/storage
    #fi
    #if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    #utils/create_split_dir.pl \
    #    /export/b{10,11,12,13}/${USER}/espnet-data/egs/wsj/asr1/dump/${train_dev}/delta${do_delta}/storage \
    #    ${feat_dt_dir}/storage
    #fi
    echo "stage 1: Dump wav files into a HDF5 file"
    false && {
    for setname in ${train_set} ${train_dev} ${train_test}; do
        mkdir -p data/${setname}_multich
        <data/${setname}/utt2spk sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' | sort -u >data/${setname}_multich/utt2spk
        <data/${setname}/text_spk1 sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' | sort -u >data/${setname}_multich/text_spk1
        <data/${setname}/text_spk2 sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' | sort -u >data/${setname}_multich/text_spk2
        <data/${setname}_multich/utt2spk utils/utt2spk_to_spk2utt.pl >data/${setname}_multich/spk2utt

        for ch in 1 2; do
            <data/${setname}/wav.scp grep "CH${ch}" | sed -r 's/^(.*?).CH[0-9](_?.*?) /\1\2 /g' >data/${setname}_multich/wav_ch${ch}.scp
        done
        mix-mono-wav-scp.py data/${setname}_multich/wav_ch*.scp >data/${setname}_multich/wav.scp
        rm -f data/${setname}_multich/wav_ch*.scp
    done

    dump_pcm.sh --cmd "$train_cmd" --nj 32 --filetype "sound.hdf5" --format flac data/${train_set}
    dump_pcm.sh --cmd "$train_cmd" --nj 32 --filetype "sound.hdf5" --format flac data/${train_dev}
    for rtask in ${recog_set}; do
        feat_recog_dir=data/${train_test} #${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump_pcm.sh --cmd "$train_cmd" --nj 32 --filetype "sound.hdf5" --format flac data/${train_test}
    done
    }
    #for setname in cv_single tr_single; do
#    for setname in train_si284; do
#        <data/${setname}/utt2spk utils/utt2spk_to_spk2utt.pl >data/${setname}/spk2utt
#    done
#    #dump_pcm.sh --cmd "$train_cmd" --nj 32 --filetype "sound.hdf5" --format flac data/tr_single
#    #dump_pcm.sh --cmd "$train_cmd" --nj 32 --filetype "sound.hdf5" --format flac data/cv_single
#    dump_pcm.sh --cmd "$train_cmd" --nj 32 --filetype "sound.hdf5" --format flac data/train_si284
    dump_pcm.sh --cmd "$train_cmd" --nj 32 --filetype "sound.hdf5" --format flac data/${train_aux_set}
fi

dict=data/lang_1char/tr_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/wsj_train_si284/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/wsj_train_si284/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"

    for setname in ${train_set} ${train_dev} ${train_test}; do
        local/data2json.sh --cmd "${train_cmd}" --nj 30 --num-spkrs 2 \
            --category "multichannel" \
            --preprocess-conf ${preprocess_config} --filetype sound.hdf5 \
            --feat data/${setname}/feats.scp --nlsyms ${nlsyms} \
            --out data/${setname}/data.json data/${setname} ${dict}
    done

    setname=tr_mix_both_anechoic_min_8k
    local/data2json.sh --cmd "${train_cmd}" --nj 30 --num-spkrs 2 \
            --category "multichannel" \
            --preprocess-conf ${preprocess_config} --filetype sound.hdf5 \
            --feat data/${setname}/feats.scp --nlsyms ${nlsyms} \
            --out data/${setname}/data.json data/${setname} ${dict}

    local/data2json.sh --cmd "${train_cmd}" --nj 30 --num-spkrs 1 \
        --category "singlespeaker" \
        --preprocess-conf ${preprocess_config} --filetype sound.hdf5 \
        --feat data/${train_aux_set}/feats.scp --nlsyms ${nlsyms} \
        --out data/${train_aux_set}/data.json data/${train_aux_set} ${dict}

    mkdir -p "data/tr_mix_both_anechoic_reverb_max_16k_singlespkr"
    concatjson.py data/tr_mix_both_reverb_max_16k/data.json data/wsj_train_si284/data.json data/tr_mix_both_anechoic_max_16k > "data/tr_mix_both_anechoic_reverb_max_16k_singlespkr/data.json"
fi
train_set=tr_mix_both_anechoic_reverb_max_16k_singlespkr

# It takes about one day. If you just want to do end-to-end ASR without LM,
# YOU CAN SKIP this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
# lmexpname=train_rnnlm_${backend}_${lmtag}
# lmexpdir=exp/${lmexpname}
lmexpdir=exp/train_rnnlm_pytorch_lm_word65000
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: LM Preparation"
    
    if [ ${use_wordlm} = true ]; then
        lmdatadir=data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data/${train_set}/text > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
                | grep -v "<" | tr "[:lower:]" "[:upper:]" > ${lmdatadir}/train_others.txt
        cut -f 2- -d" " data/${train_dev}/text > ${lmdatadir}/valid.txt
        cut -f 2- -d" " data/${train_test}/text > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        lmdict=${dict}
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
        zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
            | grep -v "<" | tr "[:lower:]" "[:upper:]" \
            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_test}/text \
                | cut -f 2- -d" " > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
    fi

    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --layer ${lm_layers} \
        --unit ${lm_units} \
        --opt ${lm_opt} \
        --sortagrad ${lm_sortagrad} \
        --batchsize ${lm_batchsize} \
        --epoch ${lm_epochs} \
        --patience ${lm_patience} \
        --maxlen ${lm_maxlen} \
        --dict ${lmdict}
fi


if [ -z ${tag} ]; then
    if [ -n "$init_from_mdl" ]; then
        expname=seed${seed}_${train_set}2c_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})_init_from_mdl${lr:+_lr$lr} #_wpd_souden
    else
        expname=seed${seed}_${train_set}2c_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})${init_frontend:+_init_frontend}${init_asr:+_init_asr}${lr:+_lr$lr} #_wpd_souden
    #expname=${train_set}_${backend}_b${blayers}_unit${bunits}_proj${bprojs}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_sampprob${samp_prob}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    fi
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${cmvn_tag}" ]; then
        expname=${expname}_${cmvn_tag}
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
#expdir=exp_revb/${expname}${multich_epochs:+_fromE$multich_epochs}_5taps_$(date +"%Y_%m_%d")
if [ -n "$test_nmics" ]; then
    chs="${test_nmics}ch_"
else
    chs=
fi

if [ -z "$resume" ]; then
    expdir=exp/${expname}${multich_epochs:+_fromE$multich_epochs}_${chs}5taps${use_vad_mask:+_vad_mask}${bf_wpe_tag:+_tag_$bf_wpe_tag}_$(date +"%Y_%m_%d")
else
    resume_dir=$(dirname "$resume")
    expdir=${resume_dir%/results}
fi

mkdir -p ${expdir}

#train_set=train_si284
#train_set=tr_spatialized_reverb_multich
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    train_opts=""
    if [ -n "${stats_file}" ]; then
        train_opts=${train_opts}" --stats-file ${stats_file}"
    fi
    if ${apply_uttmvn}; then
        train_opts=${train_opts}" --apply-uttmvn true"
    fi

    # if [ -n "${test_nmics}" ]; then
    #     datadir=data_8ch
    # else
        datadir=data
    # fi

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

	${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${datadir}/${train_set}/data.json \
        --valid-json ${datadir}/${train_dev}/data.json \
        --preprocess-conf ${preprocess_config} \
        --num-spkrs ${num_spkrs} \
        --use-WPD-frontend False \
        --load-wav-ref False \
        ${init_frontend:+--init-frontend $init_frontend} \
        ${init_asr:+--init-asr $init_asr} \
        ${init_from_mdl:+--init-from-mdl $init_from_mdl} \
        ${seed:+--seed $seed} \
        ${test_nmics:+--test-nmics $test_nmics} \
        ${multich_epochs:+--multich-epochs $multich_epochs} \
        ${use_padertorch_frontend:+--use-padertorch-frontend True} \
        ${use_vad_mask:+--use-vad-mask True} \
        ${batch_size:+--batch-size $batch_size} \
        ${lr:+--lr $lr} \
        --ctc_type 'builtin' \
        ${bf_wpe_tag:+--wpe-tag $bf_wpe_tag --beamforming-tag $bf_wpe_tag} \
        ${fbank_fs:+--fbank-fs $fbank_fs}
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    pids=() # initialize pids
    for rtask in ${train_dev} ${train_test}; do
    #for rtask in tt_spatialized_reverb_2ch; do
    #for rtask in tt_spatialized_reverb_librispeech_2ch; do
    (
        prefix=""
        if [ -n "${test_nmics}" ]; then
            prefix=${prefix}_${test_nmics}ch
        fi
        if [ -n "${test_btaps}" ]; then
            prefix=${prefix}_${test_btaps}btaps
        fi
        decode_dir=decode${prefix}_${rtask}_$(basename ${decode_config%.*})_${recog_model}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
#        decode_dir=decode${prefix}_${rtask}_$(basename ${decode_config%.*})_${recog_model}_${lmtag}
        #decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}_lm1.0_pywer
#        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}_${lmtag}
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        if [ ${lm_weight} == 0 ]; then
            recog_opts=""
        fi
        #if [ -n "${test_nmics}" ]; then
        #    feat_recog_dir=data_8ch/${rtask}  # ${dumpdir}/${rtask}/delta${do_delta}
        #else
            feat_recog_dir=data/${rtask}  # ${dumpdir}/${rtask}/delta${do_delta}
        #fi

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --num-spkrs ${num_spkrs} \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            ${seed:+--seed $seed} \
            --use-WPD-frontend False \
            ${use_vad_mask:+--use-vad-mask True} \
            ${test_btaps:+--test-btaps $test_btaps} \
            ${test_nmics:+--test-nmics $test_nmics} \
            ${lm_weight:+--lm-weight $lm_weight} \
            ${ctc_weight:+--ctc-weight $ctc_weight} \
            ${recog_opts}
        wait

         score_sclite.sh --wer true --nlsyms ${nlsyms} --num_spkrs ${num_spkrs} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
    exit 0;
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Drawing Spectrogram"
    nj=1

    setname="tt"
    for rtask in "reverb"; do
    #for rtask in ${train_dev}; do
    (
        feat_recog_dir=data/${setname}_spatialized_${rtask}_2ch  # ${dumpdir}/${rtask}/delta${do_delta}
        audio_dir=/export/c09/xkc09/asr/wsj_2mix_multi_channel/asr1/data/separated_audio/${setname}/audio2/${rtask}
        mkdir -p ${audio_dir}
        mkdir -p ${audio_dir}/../spec_visual_${setname}_${rtask}
        echo ${expdir}/${recog_model} > ${audio_dir}/readme

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/log/draw.JOB.log \
            asr_draw.py \
            --num-spkrs ${num_spkrs} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --image-dir ${audio_dir}/../spec_visual_${setname}_${rtask} \
            --enh-wspecifier "ark,scp:${audio_dir},${audio_dir}/audio.scp" \
            --enh-filetype 'sound' &
        wait

    ) &
    done
    wait
    echo "Finished"
fi

expdir=exp_revb/seed1_tr_spatialized_reverb_multich_singlespkr2c_pytorch_train_multispkr512_trans_wyz97_preprocess_init_asr_wpd_ver5_globalcmvn_5taps_2020_05_08

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Filtering Training Sampling"
    train_opts=""
    if [ -n "${stats_file}" ]; then
        train_opts=${train_opts}" --stats-file ${stats_file}"
    fi
    if ${apply_uttmvn}; then
        train_opts=${train_opts}" --apply-uttmvn true"
    fi

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/filter_samples.log \
        asr_filter_samples.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/filter_results \
        --tensorboard-dir tensorboard_filter/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json data/${train_set}/data.json \
        --valid-json data/${train_dev}/data.json \
        --preprocess-conf ${preprocess_config} \
        --num-spkrs ${num_spkrs} \
        --use-WPD-frontend False \
        ${init_frontend:+--init-frontend $init_frontend} \
        ${init_asr:+--init-asr $init_asr} \
        ${init_from_mdl:+--init-from-mdl $init_from_mdl} \
        ${seed:+--seed $seed} \
        ${multich_epochs:+--multich-epochs $multich_epochs} \
        ${use_vad_mask:+--use-vad-mask True} \
        ${lr:+--lr $lr} \
        --ctc_type 'builtin' \
        --output-json-path data/${train_set}/data_2c_prune.json

    echo "Finished"
fi
