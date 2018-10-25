#!/usr/bin/env bash
# [wf] execute separate stage

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
          code/separate_musdb.py \
            --run_directory $model_path \
            --folder "/experiment/data/raw/musdb/test/"
    fi

    if [ $USE_DOCKER -eq 0 ]; then
        model_path="pipelines/rnn-music-baseline/runs/run9"
        source activate prem
        cd ../../
        python code/separate_music.py \
            --log_dir $model_path \
            --mixture_folder "data/raw/musdb/test/"
    fi
fi