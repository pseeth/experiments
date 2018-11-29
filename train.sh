#!/bin/sh

output_folder="/exp/pipelines/runs/run0/"

cd /exp/code

python config/config.py --config_folder $output_folder/config dataset wsj 
python config/config.py --config_folder $output_folder/config dpcl_recurrent 
python config/config.py --config_folder $output_folder/config \
                            train   --training_folder /exp/data/wsj0-mix/2speakers_anechoic/wav8k/min/tr/ \
                                    --validation_folder /exp/data/wsj0-mix/2speakers_anechoic/wav8k/min/cv/ \
                                    --loss_function dpcl embedding 1.0 \
                                    --num_workers 10 \
                                    --num_epoch 100 
cd /exp

python code/train.py --train $output_folder/config/train.json \
                     --model $output_folder/config/dpcl_recurrent.json \
                     --dataset $output_folder/config/dataset.json \
                     --output_folder $output_folder