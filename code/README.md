# Conditional attractor networks

This repository contains the code for conditional attractor networks.

Dependencies:
- pytorch v0.4
- numpy, scipy
- librosa
- scaper (this installation: https://github.com/pseeth/scaper/tree/source_separation - source separation branch)
- tensorboard and tensorboardX
- jams (comes with Scaper)
- musdb
- audio_embed (https://github.com/pseeth/audio_embed - needed for Jupyter notebooks, nothing else)

./:

- create_dataset.py: Creating the dataset using the Scaper tool.
- evaluate.py: Evaluating a single track in musdb with estimates and source folders set up correctly. Computes bss_eval measures on a single track.
- loss.py: contains custom loss functions (not really used at the moment)
- mwf.py: Adapted from sigsep code, computes a multichannel wiener filter.
- test.py: Should probably rename this to separate - separates a single track in musdb with a given model, or group of models (e.g. one for vocals, one for bass, etc).
- train.py: Training the model. Has a lot of configurable arguments that get parsed. There must be a "runs" directory for this to go. The option --log_dir log_directory will save the model, tensorboard information, checkpoints, and parameter configuration into runs/log_directory.
- utils.py: Various helpful utilities (e.g. parallel execution of a function, visualizing embeddings with PCA projections, etc.
- validate.py: Validation code when running the model. Main function runs the separation on validation examples and stores images, audio, and computes the loss over the validation set and throws it onto tensorboard.

./slurm_helpers:
- baseline.slurm: Slurm configuration for a job for launching the baseline, gives it a job name of "baseline".
- conditional_deep_attractor.slurm: Identical to baseline, but with a different job name of "cond_deep_attr".
- evaluate_track.slurm: Slurm configuration for launching an evaluate job, with job name "evaluate".
- launch_eval_jobs.sh: Script for launching separate and evaluate jobs on musdb in parallel. Give it the model path and a short name for the compiled report produced by mus_eval.
- launch_jobs.sh: Launching the training jobs. Look here for examples on how to use train.py.
- separate_and_evaluate_musdb.sh: Separates all musdb tracks with a model, and then evaluates all the separations. Copies them to another directory that can be parsed by the aggregate.py script in sigsep-mus-2018-analysis (https://github.com/sigsep/sigsep-mus-2018-analysis)
- separate_track.sh: Separates a single track in musdb.

./networks:
- deep_attractor.py: Conditional deep attractor network.
- initializers.py: Networks for initializing parameters of gaussians in the conditional deep attractor network with one hot labels as input.
- mask_estimation.py: Baseline model for source separation (outputs masks directly).

./networks/clustering/
- gmm.py: Unfolded gaussian mixture model with four covariance types: "diag" (1 variance per feature per gaussian), "tied_diag" (1 variance per feature shared across gaussians), "spherical" (1 variance per gaussian for all features) "tied_spherical" (1 variance shared across all features and gaussians - reduces to KMeans with priors)
- kmeans.py: Soft k-means. Nearly identical to tied_spherical mode for GMM.
- test_clustering.ipynb: Notebook for testing the clustering algorithms on synthetic data.





