#!/usr/bin/env bash
USE_DOCKER=`cat ../DOCKER`
rm -rf data/generated

if [ $USE_DOCKER -eq 1 ]; then
    docker stop -t 0 data_generation || true
    docker run --rm --workdir=/pipeline -v `pwd`:/pipeline \
      --entrypoint python \
      --name data_generation \
      --user $UID \
      thesis scripts/create_incoherent_music.py
fi

if [ $USE_DOCKER -eq 0 ]; then
    source activate prem
    python scripts/create_incoherent_music.py
fi