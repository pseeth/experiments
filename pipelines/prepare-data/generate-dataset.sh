#!/usr/bin/env bash

docker stop -t 0 data_generation || true
docker rm data_generation || true

docker run --workdir=/pipeline -v `pwd`:/pipeline \
  --name data_generation \
  -e USERID=$UID \
  thesis scripts/create_incoherent_music.py