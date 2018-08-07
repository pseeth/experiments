#!/usr/bin/env bash
# [wf] execute train stage

mkdir -p runs
num_run=9 #$(ls runs/ | wc -l |  tr -d ' ')
run_id="run$num_run"
model_path="/experiment/pipelines/rnn-music-baseline/runs/$run_id"
echo $model_path > model_path

mkdir -p runs/$run_id
cp train.sh runs/${run_id}/${run_id}_train.sh
#git commit -am "commiting changes before $run_id"
git rev-parse HEAD > runs/${run_id}/commit_hash

#Specify GPU cloud provider using env variable?

docker run --rm --workdir=/experiment -v `pwd`/../..:/experiment \
  --runtime=nvidia \
  --name rnn-music-baseline \
  --entrypoint python \
  --ipc=host \
  thesis \
  code/train.py \
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
    --num_epochs 100 \
    --batch_size 20 \
    --initial_length .2 \
    --curriculum_learning \
    --source_labels vocals_drums_bass_other \
    --learning_rate 2e-4 \
    --projection_size 300 \
    --num_workers 10 \
    --resume \
    --sample_strategy sequential \
    --baseline \
    --reorder_sources

