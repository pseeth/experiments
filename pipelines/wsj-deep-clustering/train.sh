#!/usr/bin/env bash
# [wf] execute train stage

mkdir -p runs
num_run=$(ls runs/ | wc -l |  tr -d ' ')
run_id="run$num_run-dc-gt-1000"
dataset="2speakers_anechoic"
covariance_type="tied_spherical"

echo $model_path > model_path
USE_DOCKER=`cat ../DOCKER`

mkdir -p runs/$run_id
cp train.sh runs/${run_id}/${run_id}_train.sh
git rev-parse HEAD > runs/${run_id}/commit_hash

if [ ! -d data/wsj0-mix/2speakers_anechoic/ ]; then
    if [ $USE_DOCKER -eq 1 ]; then
        model_path="/experiment/pipelines/wsj-deep-clustering/runs/$run_id"
        docker run --rm --workdir=/experiment -v `pwd`/../..:/experiment \
          --runtime=nvidia \
          -e NVIDIA_VISIBLE_DEVICES=all \
          --name $run_id \
          --entrypoint python \
          --ipc=host \
          thesis \
          code/train.py \
            --log_dir $model_path \
            --training_folder "/experiment/data/wsj0-mix/$dataset/wav8k/min/tr/" \
            --validation_folder "/experiment/data/wsj0-mix/$dataset/wav8k/min/cv/" \
            --dataset_type wsj \
            --loss_function dc \
            --embedding_activation tanh \
            --normalize_embeddings \
            --target_type psa \
            --disable-training-stats \
            --n_fft 256 \
            --hop_length 64 \
            --hidden_size 300 \
            --num_layers 4 \
            --num_epochs 100 \
            --batch_size 40 \
            --clustering_type gmm \
            --covariance_type $covariance_type \
            --num_clustering_iterations 0 \
            --embedding_size 15 \
            --learning_rate 1e-3 \
            --initial_length 400 \
            --projection_size 0 \
            --num_workers 12 \
            --resume \
            --weight_method magnitude \
            --create_cache \
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