#!/usr/bin/env bash
# [wf] execute separate stage

#Specify GPU cloud provider using env variable?
USE_DOCKER=`cat ../DOCKER`
model_path="pipelines/wsj-deep-clustering/runs/run58-dc-bootstrap-log-confidence-mag-q-weight/"


if [ ! -d data/wsj0-mix ]; then
    if [ $USE_DOCKER -eq 1 ]; then
        prefix="/experiment"
        docker run --rm --workdir=/experiment -v `pwd`/../..:/experiment \
          --runtime=nvidia \
          --name separate-wsj \
          --entrypoint python \
          --ipc=host \
          thesis \
          code/separate_wsj.py \
            --run_directory "$prefix/$model_path" \
            --folder "/experiment/data/wsj0-mix/2speakers_anechoic/wav8k/min/tt/" \
            --device_target "cuda"
    fi
fi