#!/usr/bin/env bash
# [wf] execute train stage

mkdir -p runs
num_runs=$(ls runs/ | wc -l |  tr -d ' ')
run_id="run${num_runs}"
model_path="/experiment/pipelines/rnn-music-baseline/runs/{$run_id}/"
cp train.sh ${model_path}/${run_id}_train.sh
git commit -am "commiting changes before ${run_id}"
git rev-parse HEAD > ${model_path}/commit_hash
echo $model_path > 'model_directory'
errror out here

docker run --rm --workdir=/experiment -v `pwd`/../..:/experiment \
  --name rnn-music-baseline \
  --user $UID \
  thesis code/train.py \
    --log_dir $model_path \
    --training_folder /experiment/data/generated/musdb/train \
    --validation_folder /experiment/data/generated/musdb/validation/ \
    --loss_function l1 \
    --target_type msa \
    --activation_type sigmoid \
    --disable-training-stats \
    --n_fft 2048 \
    --hop_length 512 \
    --hidden_size 300 \
    --num_layers 4 \
    --num_epochs 10 \
    --batch_size 20 \
    --initial_length .2 \
    --source_labels vocals_drums_bass_other \
    --learning_rate 2e-4 \
    --projection_size 300 \
    --num_workers 0 \
    --overwrite \
    --sample_strategy sequential \
    --baseline \
    --reorder_sources

