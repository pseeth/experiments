#!/usr/bin/env bash
# [wf] build docker image
set -ex

docker --version
if [ $? -ne 0 ]; then
  echo "Cannot invoke docker command"
  exit 1
fi

docker build \
    --build-arg user_id=$UID \
    --build-arg user_name=`whoami` \
    -t thesis docker/

echo "1" > ../DOCKER

#nvidia-docker run --rm -t --entrypoint nvidia-smi thesis