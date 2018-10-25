#!/usr/bin/env bash
# [wf] execute separate stage

#Specify GPU cloud provider using env variable?
USE_DOCKER=`cat ../DOCKER`
model_path="pipelines/em-cluster-music/runs/run-tied_spherical-l1-dc/"

if [ ! -d data/musdb ]; then
    if [ $USE_DOCKER -eq 1 ]; then
        prefix="/experiment"
        docker run --rm --workdir=/experiment -v `pwd`/../..:/experiment \
          --runtime=nvidia \
          --name separate-em-cluster-music \
          --entrypoint python \
          --ipc=host \
          thesis \
          code/separate_musdb.py \
            --run_directory "$prefix/$model_path" \
            --folder "/experiment/data/raw/musdb/test/"
    fi

    if [ $USE_DOCKER -eq 0 ]; then
        source activate prem
        cd ../../
        python code/separate_music.py \
            --log_dir $model_path \
            --folder "data/raw/musdb/test/"
    fi
fi