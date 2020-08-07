#!/bin/bash

######################################################################
# Please make sure you run the following command before this script:
#    `local/convert2wav.sh ${wsj0_path} ${wsj1_path} YOUR_PATH`
#
# See https://github.com/Emrys365/espnet/blob/wsj1_mix_spatialized/egs/wsj1_mix_spatialized/asr1/data_simu/README.md for more details.
######################################################################

if [ $# -le 2 ]; then
  echo "Arguments should be WSJ0-2MIX directory, the mixing script path and the WSJ0 path, see ../run.sh for example."
  exit 1;
fi

. ./path.sh
find_transcripts=$KALDI_ROOT/egs/wsj/s5/local/find_transcripts.pl
normalize_transcript=$KALDI_ROOT/egs/wsj/s5/local/normalize_transcript.pl
utt2spk_to_spk2utt=$KALDI_ROOT/egs/wsj/s5/utils/utt2spk_to_spk2utt.pl

# path to the generated wsj1-2mix data
wavdir=$(realpath $1)
# path to the directory containing scripts for generating wsj1-2mix data
srcdir=$(realpath $2)
# path to the generated spatialized wsj1-2mix data
wsj0_2mix_spatialized_wavdir=$(realpath $3)
# root directory containing wsj0 and wsj1 data (by running local/convert2wav.sh)
wsj_full_wav=$(realpath $4)

# check if the wav dir exists.
for f in $wavdir/tr $wavdir/cv $wavdir/tt; do
  if [ ! -d $wavdir ]; then
    echo "Error: $wavdir is not a directory."
    exit 1;
  fi
done

# check if the script file exists.
for f in $srcdir/mix_2_spk_max_tr_mix $srcdir/mix_2_spk_max_cv_mix $srcdir/mix_2_spk_max_tt_mix; do
  if [ ! -f $f ]; then
    echo "Could not find $f.";
    exit 1;
  fi
done

data=./data
mkdir -p ${data}

for x in tr cv tt; do
  mkdir -p ${data}/$x
  cat $srcdir/mix_2_spk_max_${x}_mix | \
    awk -v dir=$wavdir/$x '{printf("%s %s/mix/%s.wav\n", $1, dir, $1)}' | \
    awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${data}/$x/wav.scp

  awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/$x/wav.scp | sort > ${data}/$x/utt2spk
  ${utt2spk_to_spk2utt} ${data}/$x/utt2spk > ${data}/$x/spk2utt
done

# transcriptions
rm -r tmp/ 2>/dev/null
mkdir -p tmp
cd tmp
cat ${wsj_full_wav}/si_tr_s.scp ${wsj_full_wav}/si_tr_s_wsj1.scp > si_tr_s.scp
cp ${wsj_full_wav}/si_et_20.scp si_et_20.scp
cp ${wsj_full_wav}/si_dt_20_wsj1.scp si_dt_20.scp

# Finding the transcript files:
for x in `ls ${wsj_full_wav}/links/`; do find -L ${wsj_full_wav}/links/$x -iname '*.dot'; done > dot_files.flist

# Convert the transcripts into our format (no normalization yet)
for f in si_tr_s si_et_20 si_dt_20; do
  cat ${f}.scp | awk '{print $1}' | ${find_transcripts} dot_files.flist > ${f}.trans1

  # Do some basic normalization steps.  At this point we don't remove OOVs--
  # that will be done inside the training scripts, as we'd like to make the
  # data-preparation stage independent of the specific lexicon used.
  noiseword="<NOISE>"
  cat ${f}.trans1 | ${normalize_transcript} ${noiseword} | sort > ${f}.txt || exit 1;
done

# change to the original path
cd ..

awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_tr_s.txt ${data}/tr/wav.scp | awk '{$2=""; print $0}' > ${data}/tr/text_spk1
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_tr_s.txt ${data}/tr/wav.scp | awk '{$2=""; print $0}' > ${data}/tr/text_spk2
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_dt_20.txt ${data}/cv/wav.scp | awk '{$2=""; print $0}' > ${data}/cv/text_spk1
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_dt_20.txt ${data}/cv/wav.scp | awk '{$2=""; print $0}' > ${data}/cv/text_spk2
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt1=lst[3]; text=txt[utt1]; print($1, text)}' tmp/si_et_20.txt ${data}/tt/wav.scp | awk '{$2=""; print $0}' > ${data}/tt/text_spk1
awk '(ARGIND==1) {txt[$1]=$0} (ARGIND==2) {split($1, lst, "_"); utt2=lst[5]; text=txt[utt2]; print($1, text)}' tmp/si_et_20.txt ${data}/tt/wav.scp | awk '{$2=""; print $0}' > ${data}/tt/text_spk2


############################################
# Generating data for spatialized wsj1-2mix
############################################
for x in tr_spatialized_anechoic_multich cv_spatialized_anechoic_multich tt_spatialized_anechoic_multich \
         tr_spatialized_reverb_multich cv_spatialized_reverb_multich tt_spatialized_reverb_multich; do
  mkdir -p ${data}/$x
  x_ori=${x%%_*}    # tr, cv, tt
  suffix=$(echo $x | rev | cut -d"_" -f2 | rev)   # anechoic, reverb
  multich_wavdir=${wsj0_2mix_spatialized_wavdir}/2speakers_${suffix}/wav16k/max/${x_ori}
  awk '{print $1}' ${data}/${x_ori}/wav.scp | \
    cut -d"_" -f 3- | \
    awk -v dir="$multich_wavdir" -v suffix="$suffix" '{printf("%s_%s %s/mix/%s.wav\n", $1, suffix, dir, $1)}' | \
    awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${data}/$x/wav.scp

  awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/$x/wav.scp | sort > ${data}/$x/utt2spk
  ${utt2spk_to_spk2utt} ${data}/$x/utt2spk > ${data}/$x/spk2utt

  # transcriptions
  paste -d " " \
    <(awk -v suffix="$suffix" '{print($1 "_" suffix)}' ${data}/${x_ori}/text_spk1) \
    <(cut -f 2- -d" " ${data}/${x_ori}/text_spk1) | sort > ${data}/${x}/text_spk1
  paste -d " " \
    <(awk -v suffix="$suffix" '{print($1 "_" suffix)}' ${data}/${x_ori}/text_spk2) \
    <(cut -f 2- -d" " ${data}/${x_ori}/text_spk2) | sort > ${data}/${x}/text_spk2
done
