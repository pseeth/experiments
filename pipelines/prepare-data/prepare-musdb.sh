#!/usr/bin/env bash
# [wf] execute setup stage

mkdir -p raw
if [ ! -d raw/musdb ]; then
    mkdir -p raw/musdb
    unzip raw/musdb18.zip -d raw/musdb/
    docker pull faroit/sigsep-mus-io
    docker run --rm -v `pwd`/raw/musdb:/data faroit/sigsep-mus-io /scripts/decode.sh
fi

mkdir -p data
if [ ! -d data/musdb ]; then
    docker run --workdir=/pipeline -v `pwd`:/pipeline \
    -e USERID=$UID \
  thesis scripts/organize_musdb.py
fi

if [ ! -d ../../data ]; then
    ln -s `pwd`/data `pwd`/../..
fi