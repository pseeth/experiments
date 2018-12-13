#!/usr/bin/env bash
rm -rf data/generated

docker stop -t 0 data_generation || true
docker run --rm --workdir=/pipeline -v `pwd`:/pipeline \
    -v `pwd`/../../data:/pipeline/data \
    --entrypoint python \
    --name data_generation \
    --user $UID \
    thesis scripts/create_incoherent_music.py