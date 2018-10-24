#!/usr/bin/env bash
# [wf] execute separate stage
run_id=9

cd ../../
code/evaluate_on_lambda.py \
    --output_directory "pipelines/rnn-music-baseline/runs/run$run_id/output" \
    --provider "aws"

aws s3 sync pipelines/rnn-music-baseline/runs/run9/output/ s3://bsseval/uploads --exclude='*' --include '*.zip'