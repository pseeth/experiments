#!/usr/bin/env bash
# [wf] execute train stage

mkdir -p runs
covariance_type="diag"
num_run=$(ls runs/ | wc -l |  tr -d ' ')
run_id="run$num_run-$covariance_type-dc-curr"

echo $model_path > model_path
USE_DOCKER=`cat ../DOCKER`

mkdir -p runs/$run_id
cp train.sh runs/${run_id}/${run_id}_train.sh
#git commit -am "commiting changes before $run_id"
git rev-parse HEAD > runs/${run_id}/commit_hash

#Specify GPU cloud provider using env variable?

if [ ! -d data/musdb ]; then
    if [ $USE_DOCKER -eq 1 ]; then
        model_path="/experiment/pipelines/em-cluster-music/runs/$run_id"
        docker run --rm --workdir=/experiment -v `pwd`/../..:/experiment \
          --runtime=nvidia \
          --name em-cluster-music \
          --entrypoint python \
          --ipc=host \
          thesis \
          code/train.py \
            --log_dir $model_path \
            --training_folder /experiment/data/generated/musdb/train \
            --validation_folder /experiment/data/generated/musdb/validation/ \
            --loss_function l1_dc \
            --target_type msa \
            --disable-training-stats \
            --n_fft 2048 \
            --hop_length 512 \
            --hidden_size 300 \
            --num_layers 4 \
            --num_epochs 100 \
            --batch_size 20 \
            --clustering_type gmm \
            --covariance_type $covariance_type \
            --num_clustering_iterations 0 \
            --embedding_size 15 \
            --initial_length .2 \
            --curriculum_learning \
            --source_labels vocals_drums_bass_other \
            --learning_rate 2e-4 \
            --projection_size 300 \
            --num_workers 10 \
            --resume \
            --sample_strategy sequential
    fi

    if [ $USE_DOCKER -eq 0 ]; then
        model_path="pipelines/em-cluster-music/runs/$run_id"
        source activate prem
        cd ../../
        python code/train.py \
            --log_dir $model_path \
            --training_folder data/generated/musdb/train \
            --validation_folder data/generated/musdb/validation/ \
            --loss_function l1 \
            --target_type msa \
            --disable-training-stats \
            --n_fft 2048 \
            --hop_length 512 \
            --hidden_size 300 \
            --num_layers 4 \
            --num_epochs 100 \
            --batch_size 20 \
            --clustering_type gmm \
            --covariance_type $covariance_type \
            --num_clustering_iterations 0 \
            --embedding_size 15 \
            --initial_length .2 \
            --curriculum_learning \
            --source_labels vocals_drums_bass_other \
            --learning_rate 2e-4 \
            --projection_size 300 \
            --num_workers 10 \
            --resume \
            --sample_strategy sequential
    fi
fi