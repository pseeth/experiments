## Experiments with conditional centroids, deep clustering, and more (to come)!

This project uses [Popper](https://github.com/systemslab/popper), a convention and CLI tool for generating reproducible papers. Install Popper with:

    pip install popper

You'll also need Docker. Install Docker following the instructions here: https://docs.docker.com/install/.

For GPU usage, you need nvidia-docker: https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)

Once those are installed, you can reproduce experiments with the following commands:

     popper run create-environment   
  
This pipeline builds a Docker image that is used for all the experiments. It also launches Tensorboard and a Jupyter server at localhost:6006 and localhost:8888, respectively.

    popper run prepare-music-data
   
This pipeline prepares all the data used in the experiments. For musdb experiments, it expects the musdb dataset (encoded as .mp4 into a zip file, as downloaded directly from [musdb](https://sigsep.github.io/datasets/musdb.html)) at ./pipelines/prepare-data/data/raw/musdb18.zip (. being the top-level directory of this repo) It then generates all the training data and places it into ./pipelines/prepare-data/data/generated/musdb. Then the data is moved to the top-level directory at ./data so it is accessible by the training scripts.

    popper run rnn-music-baseline
    
This pipeline trains a baseline model using a 4 layer BLSTM that outputs source masks directly. It's trained with L1 error between the estimated source spectrograms and the ground truth source spectrograms. Spectrograms are projected down to 300 mel filters before being passed into the BLSTM stack. The script parallelizes training across all available GPUs.

    popper run em-cluster-music
    
This pipeline trains the conditional centroid model for source separation, which maps time-frequency points in a spectrogram to an embedding space. A GMM with conditional starting points (conditioned on class) is unfolded on the embedding space, assigning each time-frequency point to one of the components of the GMM. The posterior of each time-frequency point (likelihood for point for one component of GMM / sum of likelihoods for point across all components of GMM) is used as the soft time frequency mask. Like in the baseline, spectrograms are projected down to 300 mel filters, and learning of the full-resolution soft mask is done through the inverse projection.

### Viewing logs
To easily view logs, you can use the logs.sh script that is in the pipelines folder. It runs "tail -f" on every available log file, with the most recently modified at the bottom.
