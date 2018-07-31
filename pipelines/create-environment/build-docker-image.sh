#!/usr/bin/env bash
# [wf] execute setup.sh stage
set -ex

docker --version
if [ $? -ne 0 ]; then
  echo "Cannot invoke docker command"
  exit 1
fi

docker build -t thesis docker/
