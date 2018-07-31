#!/usr/bin/env bash
# [wf] execute setup stage

mkdir -p raw
mkdir -p raw/musdb
#unzip raw/musdb18.zip
tar xvf raw/musdb18.zip -C raw/musdb
docker pull faroit/sigsep-mus-io
docker run --rm -v `pwd`/raw/musdb:/data faroit/sigsep-mus-io /scripts/decode.sh