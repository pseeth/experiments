#!/bin/sh

output_folder="/exp/pipelines/runs/music/"

cd /exp/code

python config/config.py --config_folder $output_folder/config dataset scaper \
                                                --n_fft 1024 \
                                                --hop_length 512 \
                                                --sample_rate 44100 \
                                                --group_sources bass drums other
python config/config.py --config_folder $output_folder/config dpcl_recurrent --sample_rate 44100 --num_frequencies 512
python config/config.py --config_folder $output_folder/config \
                            train   --training_folder /exp/data/generated/musdb/train/ \
                                    --validation_folder /exp/data/generated/musdb/validation/ \
                                    --loss_function dpcl embedding 1.0 \
                                    --num_workers 10 \
                                    --num_epoch 100 
cd /exp

python code/train.py --train $output_folder/config/train.json \
                     --model $output_folder/config/dpcl_recurrent.json \
                     --dataset $output_folder/config/dataset.json \
                     --output_folder $output_folder     
