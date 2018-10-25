#!/usr/bin/env bash
# [wf] execute separate stage
run_id=9

#Specify GPU cloud provider using env variable?
USE_DOCKER=`cat ../DOCKER`

if [ ! -d data/musdb ]; then
    if [ $USE_DOCKER -eq 1 ]; then
        model_path="/experiment/pipelines/rnn-music-baseline/runs/run9"
        docker run --rm --workdir=/experiment -v `pwd`/../..:/experiment \
          --runtime=nvidia \
          --name separate-rnn-music-baseline \
          --entrypoint python \
          --ipc=host \
          thesis \
          code/evaluate_on_cloud.py \
            --output_directory "/experiment/pipelines/rnn-music-baseline/runs/run$run_id/output" \
            --provider "aws"
    fi

    if [ $USE_DOCKER -eq 0 ]; then
        model_path="pipelines/em-cluster-music/runs/run$run_id"
        source activate prem
        cd ../../
        python code/evaluate_on_cloud.py \
            --output_directory "pipelines/rnn-music-baseline/runs/run$run_id/output" \
            --provider "aws"
    fi
fi

cd ../../


aws s3 sync pipelines/rnn-music-baseline/runs/run9/output/ s3://bsseval/uploads --exclude='*' --include '*.zip'