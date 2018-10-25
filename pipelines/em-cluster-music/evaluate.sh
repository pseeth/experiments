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
          --name evaluate-em-cluster-music \
          --entrypoint python \
          --ipc=host \
          thesis \
          code/evaluate_on_cloud.py \
            --output_directory "$prefix/$model_path/output"
    fi

    if [ $USE_DOCKER -eq 0 ]; then
        source activate prem
        cd ../../
        python code/evaluate_on_cloud.py \
            --output_directory "$model_path/output"
    fi
fi

cd ../../

aws s3 sync "$model_path/output" "s3://bsseval/uploads" --exclude='*' --include '*.zip'

while [ $(aws s3 ls s3://bsseval/results/ | wc -l) -le 49 ]
do
    echo "Waiting for evaluation jobs to finish"
    sleep 10
done

mkdir -p "$model_path/results"
aws s3 sync "s3://bsseval/results/" "$model_path/results"
aws s3 rm "s3://bsseval/results/" --recursive