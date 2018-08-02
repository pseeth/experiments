#!/usr/bin/env bash

docker stop -t 0 data_generation || true
rm -rf data/generated

docker run --rm --workdir=/pipeline -v `pwd`:/pipeline \
  --name data_generation \
  --user $UID \
  thesis scripts/create_incoherent_music.py