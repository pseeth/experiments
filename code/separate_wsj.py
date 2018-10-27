import torch
import utils
import numpy as np
from sklearn.cluster import KMeans
from wsj_dataset import WSJ0
from gmm_spatial_clustering import gmm_spatial_clustering
import argparse
import os
from scipy.io import wavfile

def separate(mixture, model, params, device):
    labels = ['s%d' % i for i in range(1, params['num_attractors'] + 1)]
    estimates = {}

    mix = mixture
    if (len(mix.shape) > 1):
        mix = mixture[:, 0]
    _, mix = utils.mask_mixture(1, mix, params['n_fft'], params['hop_length'])
    log_spec = utils.transform(mix, params['n_fft'], params['hop_length'])
    silence_mask = log_spec > -60
    log_spec = utils.whiten(log_spec)

    with torch.no_grad():
        input_data = torch.from_numpy(log_spec).unsqueeze(0).requires_grad_().to(device)
        if 'DeepAttractor' in str(model):
            with torch.no_grad():
                masks, _, embedding, _ = model(input_data, one_hots=None)

            clusterer = KMeans(n_clusters=params['num_attractors'])
            embedding_ = embedding.squeeze(0).cpu().data.numpy()
            clusterer.fit(embedding_[silence_mask.flatten()])
            assignments = clusterer.predict(embedding_)
            assignments = assignments.reshape((masks.shape[1], masks.shape[2]))

    for i, label in enumerate(labels):
        mask = (assignments == i).T.astype(float)
        source, mix = utils.mask_mixture(mask, mix, params['n_fft'], params['hop_length'])
        estimates[label] = source

    return estimates

def separate_spatial(mixture, params):
    labels = ['s1', 's2']
    estimates = {}

    mix_log_magnitude, mix_stft, _ = utils.stereo_transform(mixture, params['n_fft'], params['hop_length'])
    assignments, scores = gmm_spatial_clustering(mix_stft.swapaxes(0, 1),
                                                 mix_log_magnitude.swapaxes(0, 1),
                                                 params['sample_rate'],
                                                 params['num_attractors'],
                                                 params['n_fft'])
    for i, label in enumerate(labels):
        mask = assignments[:, :, i].T
        source, mix = utils.mask_mixture(mask, mixture[0], params['n_fft'], params['hop_length'])
        estimates[label] = source

    return estimates

def save_estimates(estimates, target_directory):
    os.makedirs(os.path.join(target_directory, 'estimates'), exist_ok=True)
    maxv = np.iinfo(np.int16).max
    for label in estimates:
        estimates[label] = estimates[label].T
        save_path = os.path.join(target_directory, 'estimates', label + '.wav')
        estimates[label] = (estimates[label] * maxv).astype(np.int16)
        wavfile.write(save_path, target_sr, estimates[label].T)

def separate_directory(run, mixture_folder, device_target):
    os.makedirs(run, exist_ok=True)
    mix_directory = os.listdir(os.path.join(mixture_folder, 'mix'))
    num_speakers = len([x for x in os.listdir(mixture_folder) if 's' in x])
    spk_directories = [os.path.join(mixture_folder, 's%d' % i) for i in range(num_speakers)]
    #mixes = [os.path.join()
    print(mix_directory, spk_directories)

    # mix, sr = utils.load_audio(audio_file)


    # if run == 'spatial':
    #     channel_indices = np.arange(mix.shape[0])
    #     np.random.shuffle(channel_indices)
    #     channel_indices = channel_indices[:2]
    #     estimates = separate_spatial(mix[channel_indices], params)
    # else:
    #     model, params, device = utils.load_model(run, device_target='cuda')
    #     estimates = separate(mix[0], model, params, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_directory")
    parser.add_argument("--mixture_folder")
    args = parser.parse_args()

    run_directories = ['/pipeline/pipelines/wsj-deep-clustering/runs/spatial',
                       '/pipeline/pipelines/wsj-deep-clustering/runs/run25-dc-ground-truth/',
                       '/pipeline/pipelines/wsj-deep-clustering/runs/run26-dc-bootstrap-confidence-threshold//',
                       '/pipeline/pipelines/wsj-deep-clustering/runs/run33-dc-bootstrap-confidence-mag-weight'
                       ]

    separate_directory(args.run_directory, args.mixture_folder, 'cpu')