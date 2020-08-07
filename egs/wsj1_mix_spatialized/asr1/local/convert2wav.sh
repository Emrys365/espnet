#!/bin/bash

. ./path.sh
. ./cmd.sh

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
  echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
  exit 1;
fi

ndx2flist=$KALDI_ROOT/egs/wsj/s5/local/ndx2flist.pl
flist2scp=$KALDI_ROOT/egs/wsj/s5/local/flist2scp.pl

WSJ0=$1
WSJ1=$2
dir=$3

mkdir -p ${dir}
cd ${dir}

rm -r links/ 2>/dev/null
mkdir -p links
ln -s ${WSJ0}/??-{?,??}.? links
ln -s ${WSJ1}/??-{?,??}.? links

# Do some basic checks that we have what we expected.
if [ ! -d links/11-13.1 ]; then
  echo "WSJ0 directory may be in a noncompatible form."
  exit 1;
fi

for disk in 11-1.1 11-2.1 11-3.1; do
  for spk in `ls links/${disk}/wsj0/si_tr_s`; do
    ls links/${disk}/wsj0/si_tr_s/$spk | grep wv1 | \
      awk -v pwd=$PWD -v disk=$disk -v spk=$spk '{printf("%s/links/%s/wsj0/si_tr_s/%s/%s\n", pwd, disk, spk, $1)}'
  done
done | sort > si_tr_s.flist

for disk in 13-1.1 13-3.1 13-5.1 13-6.1 13-11.1 13-19.1; do
  for spk in `ls links/${disk}/wsj1/si_tr_s`; do
    ls links/${disk}/wsj1/si_tr_s/$spk | grep wv1 | \
      awk -v pwd=$PWD -v disk=$disk -v spk=$spk '{printf("%s/links/%s/wsj1/si_tr_s/%s/%s\n", pwd, disk, spk, $1)}'
  done
done | sort > si_tr_s_wsj1.flist

disk=11-14.1;
for spk in `ls links/${disk}/wsj0/si_et_05`; do
  ls links/${disk}/wsj0/si_et_05/$spk | grep wv1 | \
    awk -v pwd=$PWD -v disk=$disk -v spk=$spk '{printf("%s/links/%s/wsj0/si_et_05/%s/%s\n", pwd, disk, spk, $1)}'
done | sort > si_et_05.flist

disk=11-14.1;
for spk in `ls links/${disk}/wsj0/si_et_20`; do
  ls links/${disk}/wsj0/si_et_20/$spk | grep wv1 | \
    awk -v pwd=$PWD -v disk=$disk -v spk=$spk '{printf("%s/links/%s/wsj0/si_et_20/%s/%s\n", pwd, disk, spk, $1)}'
done | sort > si_et_20.flist

disk=11-6.1;
for spk in `ls links/${disk}/wsj0/si_dt_05`; do
  ls links/${disk}/wsj0/si_dt_05/$spk | grep wv1 | \
    awk -v pwd=$PWD -v disk=$disk -v spk=$spk '{printf("%s/links/%s/wsj0/si_dt_05/%s/%s\n", pwd, disk, spk, $1)}'
done | sort > si_dt_05.flist

disk=13-16.1;
for spk in `ls links/${disk}/wsj1/si_dt_20`; do
  ls links/${disk}/wsj1/si_dt_20/$spk | grep wv1 | \
    awk -v pwd=$PWD -v disk=$disk -v spk=$spk '{printf("%s/links/%s/wsj1/si_dt_20/%s/%s\n", pwd, disk, spk, $1)}'
done | sort > si_dt_20_wsj1.flist

for f in si_tr_s si_et_05 si_et_20 si_dt_05 si_tr_s_wsj1 si_dt_20_wsj1; do
  ${flist2scp} ${f}.flist | sort > ${f}.scp
done

# Create scp's with wav's. (the wv1 in the distribution is not really wav, it is sph.)
awk -v dir=${dir} 'BEGIN{print("#!/bin/bash")}{len=split($2, lst, "/"); spk_wav=lst[len-1]"/"lst[len]; gsub(/\.wv1/, "", spk_wav); printf("'$sph2pipe' -f wav %s %s/wsj0/si_tr_s/%s.wav \n",  $2, dir, spk_wav);}' < si_tr_s.scp > si_tr_s_wav.sh
awk -v dir=${dir} 'BEGIN{print("#!/bin/bash")}{len=split($2, lst, "/"); spk_wav=lst[len-1]"/"lst[len]; gsub(/\.wv1/, "", spk_wav); printf("'$sph2pipe' -f wav %s %s/wsj0/si_dt_05/%s.wav \n", $2, dir, spk_wav);}' < si_dt_05.scp > si_dt_05_wav.sh
awk -v dir=${dir} 'BEGIN{print("#!/bin/bash")}{len=split($2, lst, "/"); spk_wav=lst[len-1]"/"lst[len]; gsub(/\.wv1/, "", spk_wav); printf("'$sph2pipe' -f wav %s %s/wsj0/si_et_05/%s.wav \n", $2, dir, spk_wav);}' < si_et_05.scp > si_et_05_wav.sh
awk -v dir=${dir} 'BEGIN{print("#!/bin/bash")}{len=split($2, lst, "/"); spk_wav=lst[len-1]"/"lst[len]; gsub(/\.wv1/, "", spk_wav); printf("'$sph2pipe' -f wav %s %s/wsj0/si_et_20/%s.wav \n", $2, dir, spk_wav);}' < si_et_20.scp > si_et_20_wav.sh
awk -v dir=${dir} 'BEGIN{print("#!/bin/bash")}{len=split($2, lst, "/"); spk_wav=lst[len-1]"/"lst[len]; gsub(/\.wv1/, "", spk_wav); printf("'$sph2pipe' -f wav %s %s/wsj1/si_et_05/%s.wav \n", $2, dir, spk_wav);}' < si_tr_s_wsj1.scp > si_tr_s_wsj1_wav.sh
awk -v dir=${dir} 'BEGIN{print("#!/bin/bash")}{len=split($2, lst, "/"); spk_wav=lst[len-1]"/"lst[len]; gsub(/\.wv1/, "", spk_wav); printf("'$sph2pipe' -f wav %s %s/wsj1/si_et_05/%s.wav \n", $2, dir, spk_wav);}' < si_dt_20_wsj1.scp > si_dt_20_wsj1_wav.sh

for f in si_tr_s si_dt_05 si_et_05 si_et_20 si_tr_s_wsj1 si_dt_20_wsj1; do
  awk '(NR>1){print $NF}' ${f}_wav.sh | \
    awk '{len=split($1, lst, "/"); dir=lst[len-3]"/"lst[len-2]"/"lst[len-1]; print dir}' | \
    sort -u > ${f}_wav.dir
  for i in `cat ${f}_wav.dir`; do mkdir -p $i; done
  rm ${f}_wav.dir
  chmod +x ${f}_wav.sh
done

rm -r log 2>/dev/null
mkdir -p log

pid=()
for script_name in si_tr_s_wav si_dt_05_wav si_et_05_wav si_et_20_wav si_tr_s_wsj1_wav si_dt_20_wsj1_wav; do
  (
    $train_cmd log/${script_name}.log \
      ./${script_name}.sh
  ) &
  pids+=($!)
done

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
