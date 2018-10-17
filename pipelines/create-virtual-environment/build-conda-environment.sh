#!/bin/bash

source activate prem
if [ $? -eq 1 ]; then
    conda create python=3.6 -y --name prem
    source activate prem
fi

conda install -y ffmpeg
conda install -y -c conda-forge sox

# Install required packages.
pip install -r requirements.txt

git clone https://github.com/pseeth/scaper
cd scaper
git checkout source_separation
pip install -e .
cd ..

pip install -U git+https://github.com/pseeth/audio_embed.git
conda install -y pytorch torchvision -c pytorch

echo "0" > ../DOCKER