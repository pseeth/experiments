import torch
import utils
import numpy as np
from sklearn.cluster import KMeans
from gmm_spatial_clustering import gmm_spatial_clustering
import argparse
import os
from scipy.io import wavfile
from tqdm import trange
import pickle

def separate(mixture, model, params, device):
    labels = ['s%d' % i for i in range(1, params['num_attractors'] + 1)]
    estimates = {}

    mix = mixture
    if (len(mix.shape) > 1):
        mix = mixture[:, 0]
    _, mix = utils.mask_mixture(1, mix, params['n_fft'], params['hop_length'])
    log_spec = utils.transform(mix, params['n_fft'], params['hop_length'])
    silence_mask = log_spec > -25
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

    return estimates, scores

def save_estimates(estimates, target_directory, sr):
    os.makedirs(os.path.join(target_directory, 'estimates'), exist_ok=True)
    maxv = np.iinfo(np.int16).max
    for label in estimates:
        estimates[label] = estimates[label].T
        save_path = os.path.join(target_directory, 'estimates', label + '.wav')
        estimates[label] = (estimates[label] * maxv).astype(np.int16)
        wavfile.write(save_path, sr, estimates[label].T)

def save_references(references, target_directory, sr):
    os.makedirs(os.path.join(target_directory, 'references'), exist_ok=True)
    maxv = np.iinfo(np.int16).max
    for label in references:
        references[label] = references[label].T
        save_path = os.path.join(target_directory, 'references', label + '.wav')
        references[label] = (references[label] * maxv).astype(np.int16)
        wavfile.write(save_path, sr, references[label].T)

def save_scores(scores, target_directory):
    os.makedirs(os.path.join(target_directory, 'scores'), exist_ok=True)
    save_path = os.path.join(target_directory, 'scores.p')
    with open(save_path, 'wb') as f:
        pickle.dump(scores, f)

def load_references_and_mix(mixture_folder, spk_directories, mix_file):
    references = {}
    mixture, sr = utils.load_audio(os.path.join(mixture_folder, 'mix', mix_file))
    for spk_directory in spk_directories:
        reference = os.path.join(mixture_folder, spk_directory, mix_file)
        references[spk_directory.split('/')[-1]] = utils.load_audio(reference)[0][0]
    return mixture, references, sr

def separate_directory(run, mixture_folder, device_target):
    os.makedirs(run, exist_ok=True)
    mix_files = os.listdir(os.path.join(mixture_folder, 'mix'))
    num_speakers = len([x for x in os.listdir(mixture_folder) if 's' in x])
    spk_directories = [os.path.join(mixture_folder, 's%d' % i) for i in range(1, num_speakers+1)]
    #mixes = [os.path.join()
    progress_bar = trange(len(mix_files))
    model, params, device = utils.load_model(run, device_target)

    for i in progress_bar:
        mix_file = mix_files[i]
        target_directory = os.path.join(run, 'output', mix_file)
        progress_bar.set_description(mix_file)

        mix, references, sr = load_references_and_mix(mixture_folder, spk_directories, mix_file)
        if 'spatial' in run:
            right_channel = np.random.randint(1, mix.shape[0])
            channel_indices = [0, right_channel]
            estimates, scores = separate_spatial(mix[channel_indices], params)
            save_scores(scores, target_directory)
        else:
            estimates = separate(mix[0], model, params, device)

        save_estimates(estimates, target_directory, sr)
        save_references(references, target_directory, sr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_directory")
    parser.add_argument("--folder")
    parser.add_argument("--device_target", default='cuda')
    args = parser.parse_args()
    try:
        separate_directory(args.run_directory, args.folder, args.device_target)
    except:
        pass