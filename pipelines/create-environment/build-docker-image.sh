#!/usr/bin/env bash
# [wf] build docker image
set -ex

docker --version
if [ $? -ne 0 ]; then
  echo "Cannot invoke docker command"
  exit 1
fi

docker build -t thesis docker/

#nvidia-docker run --rm -t --entrypoint nvidia-smi thesis