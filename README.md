# Experiments with conditional centroids, deep clustering, and more!

This project uses [Popper](https://github.com/systemslab/popper), a convention and CLI tool for generating reproducible papers. Install Popper with:

    pip install popper

You'll also need Docker. Install Docker following the instructions here: https://docs.docker.com/install/.

For GPU usage, you need nvidia-docker: https://github.com/NVIDIA/nvidia-docker

Once those are installed, you can reproduce experiments with the following commands:

     popper run create-environment   
  
This pipeline builds a Docker image that is used for all the experiments. It also launches Tensorboard and a Jupyter server at localhost:6006 and localhost:8888, respectively.

    popper run prepare-data
   
This pipeline prepares all the data used in the experiments. For musdb experiments, it expects the musdb dataset (encoded as .mp4 into a zip file, as downloaded directly from [musdb](https://sigsep.github.io/datasets/musdb.html)) at data/raw/musdb18.zip inside the prepare-data pipeline directory. It then generates all the training data.

    popper run rnn-music-baseline
    
This pipeline trains a baseline model using a 4 layer BLSTM that outputs source masks directly. It's trained with L1 error between the estimated source spectrograms and the ground truth source spectrograms. Spectrograms are projected down to 300 mel filters before being passed into the BLSTM stack. The script parallelizes training across all available GPUs.
