#!/bin/sh

output_folder="/exp/pipelines/runs/music_mel_300/"

cd /exp/code

python config/config.py --config_folder $output_folder/config dataset scaper \
                                                --n_fft 2048 \
                                                --hop_length 512 \
                                                --sample_rate 44100 \
                                                --group_sources bass drums other \
                                                --cache /media/cache/
python config/config.py --config_folder $output_folder/config dpcl_recurrent \
                                                --sample_rate 44100 \
                                                --num_frequencies 1024 \
                                                --num_mels 300 \
                                                --embedding_activations sigmoid unitnorm
python config/config.py --config_folder $output_folder/config \
                            train   --training_folder /exp/data/generated/musdb/train/ \
                                    --validation_folder /exp/data/generated/musdb/validation/ \
                                    --loss_function dpcl embedding 1.0 \
                                    --num_workers 12 \
                                    --num_epoch 100 \
                                    --batch_size 40
cd /exp

python code/train.py --train $output_folder/config/train.json \
                     --model $output_folder/config/dpcl_recurrent.json \
                     --dataset $output_folder/config/dataset.json \
                     --output_folder $output_folder     
