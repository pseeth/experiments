import torch
import numpy as np
from tqdm import trange
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import shutil
import inspect
import librosa
from scipy.io import wavfile
from networks import *
import json
import os

def magnitude_weights(magnitude):
    weights = magnitude / (np.sum(magnitude))
    weights *= (magnitude.shape[0] * magnitude.shape[1])
    return weights

def source_activity_weights(source_log_magnitudes, threshold=-40):
    above_threshold = (source_log_magnitudes - np.max(source_log_magnitudes)) > threshold
    return np.max(above_threshold, axis=-1).astype(np.float32)

def source_magnitude_weights(source_magnitudes):
    shape = source_magnitudes.shape
    num_sources = shape[-1]
    weights = source_magnitudes / (np.sum(source_magnitudes.reshape((-1, num_sources)), axis=0))
    weights *= shape[0]*shape[1]
    weights = np.max(weights, axis=-1)
    return weights

def threshold_weights(log_magnitude, threshold=-40):
    return ((log_magnitude - np.max(log_magnitude)) > threshold).astype(np.float32)

def load_audio(file_path):
    rate, audio = wavfile.read(file_path)
    if len(audio.shape) == 1:
        audio = np.expand_dims(audio, axis=-1)
    audio = audio.astype(np.float32, order='C') / 32768.0
    return audio.T, rate

def pad_packed_collate(batch, target_length=400, num_channels=1):
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
    for i in range(len(sorted_batch)):
        length = sorted_batch[i][0].shape[1]
        if length > target_length:
            offset = np.random.randint(0, length - target_length)
        else:
            offset = 0
        for j in range(len(sorted_batch[i]) - 1):
            sorted_batch[i] = list(sorted_batch[i])
            pad_length = max(target_length - length, 0)
            pad_tuple = [(0, 0) for k in range(len(sorted_batch[i][j].shape))]
            pad_tuple[1] = (0, pad_length)
            sorted_batch[i][j] = np.pad(sorted_batch[i][j], pad_tuple, mode='constant')
            if num_channels == 1:
                sorted_batch[i][j] = sorted_batch[i][j][:, offset:offset+target_length, 0]
            else:
                sorted_batch[i][j] = sorted_batch[i][j][:, offset:offset + target_length, :num_channels]
            sorted_batch[i] = tuple(sorted_batch[i])

    zipped_batch = list(zip(*sorted_batch))
    zipped_batch = [np.stack(z, axis=0) for z in zipped_batch]
    spectrogram, magnitude_spectrogram, source_spectrograms, source_ibms, weights, one_hots = \
        (torch.from_numpy(z).transpose(2, 1) for z in zipped_batch)

    return spectrogram, magnitude_spectrogram, source_spectrograms, source_ibms, weights, one_hots

def load_class_from_params(params, class_func):
    arguments = inspect.getfullargspec(class_func).args[1:]
    if 'input_size' not in params and 'input_size' in arguments:
        params['input_size'] = int(params['n_fft']/2 + 1)
    if 'num_sources' not in params and 'num_sources' in arguments:
        params['num_sources'] = params['num_attractors']
    filtered_params = {p: params[p] for p in params if p in arguments}
    return class_func(**filtered_params)

def load_model(run_directory, device_target='cuda'):
    with open(os.path.join(run_directory, 'params.json'), 'r') as f:
        params = json.load(f)

    model = None
    device = None

    if 'spatial' not in run_directory:
        saved_model_path = os.path.join(run_directory, 'checkpoints/latest.h5')
        device = torch.device('cuda', 1) if device_target == 'cuda' else torch.device('cpu')
        class_func = MaskEstimation if 'baseline' in run_directory else DeepAttractor
        model = load_class_from_params(params, class_func).to(device)

        model.eval()
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        show_model(model)

    return model, params, device

def stereo_transform(data, n_fft, hop_length):
    n = data.shape[-1]
    data = librosa.util.fix_length(data, n + n_fft // 2, axis=-1)
    stft = np.stack([librosa.stft(data[ch], n_fft=n_fft, hop_length=hop_length)
                     for ch in range(data.shape[0])], axis=-1)
    log_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return log_spec, stft, n

def transform(data, n_fft, hop_length):
    n = len(data)
    data = librosa.util.fix_length(data, n + n_fft // 2)
    stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length).T
    log_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return log_spec

def whiten(data):
    data -= data.mean()
    data /= data.std() + 1e-7
    return data

def mask_mixture(source_mask, mix, n_fft, hop_length):
    n = len(mix)
    mix = librosa.util.fix_length(mix, n + n_fft // 2)
    mix_stft = librosa.stft(mix, n_fft=n_fft, hop_length=hop_length)
    mix = librosa.istft(mix_stft, hop_length=hop_length, length=n)
    masked_mix = mix_stft * source_mask
    source = librosa.istft(masked_mix, hop_length=hop_length, length=n)
    return source, mix


def project_embeddings(embedding, num_dimensions=3, t=0.0, fig=None, ax=None, bins=None, gridsize=50):
    """
    Does a PCA projection of the embedding space
    Args:
        num_dimensions:
    Returns:
    """
    
    if (embedding.shape[-1] > 2):
        transform = PCA(n_components=num_dimensions)
        output_transform = transform.fit_transform(embedding)
    else:
        output_transform = embedding
        
    if num_dimensions == 2:
        xmin = output_transform[:, 0].min() - t
        xmax = output_transform[:, 0].max() + t
        ymin = output_transform[:, 1].min() - t
        ymax = output_transform[:, 1].max() + t

        plt.hexbin(output_transform[:, 0], output_transform[:, 1],  bins=bins, gridsize=gridsize)
        if ax is None:
            ax = fig.add_subplot(111)
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xlabel('PCA dim 1')
        plt.ylabel('PCA dim 2')
        plt.title('Embedding visualization')
    elif num_dimensions == 3:
        result=pd.DataFrame(output_transform, columns=['PCA%i' % i for i in range(3)])
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], cmap="Set2_r", s=60)

        # make simple, bare axis lines through space:
        xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
        ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
        yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
        ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
        zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
        ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

        # label the axes
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("Embedding visualization")

    
    return output_transform, ax

def compute_statistics_on_dataset(dset_loader, device):
    means = []
    stds = []
    progress_bar = trange(len(dset_loader))
    progress_bar.set_description('Computing statistics')
    for i, input_data in enumerate(dset_loader):
        input_data = input_data[0].to(device)
        mean = input_data.mean()
        std = input_data.std()
        means.append(mean.item())
        stds.append(std.item())
        progress_bar.update(1)
    return float(np.mean(means)), float(np.mean(stds))

def show_model(model):
    print(model)
    num_parameters = 0
    for p in model.parameters():
        if p.requires_grad:
            num_parameters += np.cumprod(p.size())[-1]
    print('Number of parameters: %d' % num_parameters)

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-3] + '_best.h5')

def move_optimizer(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)