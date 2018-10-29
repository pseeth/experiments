#!/usr/bin/env bash
# [wf] execute evaluate stage

#Specify GPU cloud provider using env variable?
USE_DOCKER=`cat ../DOCKER`
model_path="pipelines/wsj-deep-clustering/runs/run58-dc-bootstrap-log-confidence-mag-q-weight"
tag="dry"

if [ ! -d data/wsj0-mix ]; then
    if [ $USE_DOCKER -eq 1 ]; then
        prefix="/experiment"
        docker run --rm --workdir=/experiment -v `pwd`/../..:/experiment \
          -v /home/prem/.aws:/home/prem/.aws \
          --runtime=nvidia \
          --name evaluate-wsj \
          --entrypoint python \
          --ipc=host \
          thesis \
          code/evaluate_on_cloud.py \
            --output_directory "$prefix/$model_path/output" \
            --bucket "bsseval" \
            --upload_folder "uploads/permute/si_sdr"
    fi
fi

cd ../../

aws s3 sync "$model_path/output" "s3://bsseval/uploads/permute/si_sdr/" --exclude='*' --include '*.zip'

while [ $(aws s3 ls s3://bsseval/results/ | wc -l) -le 2999 ]
do
    echo "Waiting for evaluation jobs to finish"
    sleep 10
done

mkdir -p "$model_path/results/$tag"
aws s3 sync "s3://bsseval/results/" "$model_path/results/$tag"
aws s3 rm "s3://bsseval/results/" --recursive