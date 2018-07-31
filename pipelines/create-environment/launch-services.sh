#!/usr/bin/env bash
# [wf] execute generate-figures stage
TENSORBOARD_HOST_PORT=6006
JUPYTER_HOST_PORT=8888

./stop-services.sh

docker run --workdir=/pipeline -v `pwd`/..:/pipeline \
    -p 127.0.0.1:$TENSORBOARD_HOST_PORT:6006 -d \
    --entrypoint tensorboard \
    --name tensorboard \
  thesis --logdir .

docker run --workdir=/pipeline -v `pwd`/..:/pipeline \
    -p 127.0.0.1:$JUPYTER_HOST_PORT:8888 -d \
    --entrypoint jupyter \
    --name jupyter \
  thesis lab --ip=0.0.0.0 --allow-root