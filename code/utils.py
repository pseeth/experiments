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

def weight_by_magnitude(magnitude):
    weights = magnitude / np.sum(magnitude)
    weights *= (magnitude.shape[0] * magnitude.shape[1])
    return np.sqrt(weights)

def source_activity_weights(source_magnitudes, threshold=-40):
    log_magnitude = 20 * np.log10(source_magnitudes + 1e-8)
    above_threshold = (log_magnitude - np.max(log_magnitude)) > threshold
    return np.max(above_threshold, axis=-1).astype(np.float32)

def pad_packed_collate(batch):
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
    max_length = sorted_batch[0][0].shape[1]
    for i in range(len(sorted_batch)):
        for j in range(len(sorted_batch[i]) - 1):
            sorted_batch[i] = list(sorted_batch[i])
            length = sorted_batch[i][j].shape[1]
            pad_length = max_length - length
            pad_tuple = [(0, 0) for k in range(len(sorted_batch[i][j].shape))]
            pad_tuple[1] = (0, pad_length)
            sorted_batch[i][j] = np.pad(sorted_batch[i][j], pad_tuple, mode='constant')
            sorted_batch[i] = tuple(sorted_batch[i])
    zipped_batch = list(zip(*sorted_batch))
    zipped_batch = [np.stack(z, axis=0) for z in zipped_batch]
    spectrogram, magnitude_spectrogram, source_spectrograms, source_ibms, one_hots = \
        (torch.from_numpy(z) for z in zipped_batch)

    return spectrogram, magnitude_spectrogram, source_spectrograms, source_ibms, one_hots

def mask_mixture(mask, mix, n_fft, hop_length):
    n = len(mix)
    mix = librosa.util.fix_length(mix, n + n_fft // 2)
    mix_stft = librosa.stft(mix, n_fft=n_fft, hop_length=hop_length)
    masked_mix = mix_stft * mask
    source = librosa.istft(masked_mix, hop_length=hop_length, length=n)
    return source

def load_class_from_params(params, class_func):
    arguments = inspect.getfullargspec(class_func).args[1:]
    if 'input_size' not in params and 'input_size' in arguments:
        params['input_size'] = int(params['n_fft']/2 + 1)
    if 'num_sources' not in params and 'num_sources' in arguments:
        params['num_sources'] = params['num_attractors']
    filtered_params = {p: params[p] for p in params if p in arguments}
    return class_func(**filtered_params)

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