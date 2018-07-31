#!/usr/bin/env bash

docker stop -t 0 tensorboard || true
docker rm tensorboard || true
docker stop -t 0 jupyter  || true
docker rm jupyter