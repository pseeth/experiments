#!/usr/bin/env bash
# [wf] execute setup stage

mkdir -p data
mkdir -p data/raw
if [ ! -d data/raw/musdb ]; then
    mkdir -p data/raw/musdb
    #unzip data/raw/musdb18.zip -d data/raw/musdb
    tar xvf data/raw/musdb18.zip -C data/raw/musdb
    docker pull faroit/sigsep-mus-io
    docker run --rm -v `pwd`/data/raw/musdb:/data faroit/sigsep-mus-io /scripts/decode.sh
    docker rmi faroit/sigsep-mus-io
fi

rm data/raw/musdb/train/*.mp4 data/raw/musdb/test/*.mp4

if [ ! -d data/musdb ]; then
    docker run --rm --workdir=/pipeline -v `pwd`:/pipeline \
    --user $UID \
  thesis scripts/organize_musdb.py
fi