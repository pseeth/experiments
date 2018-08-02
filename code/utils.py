import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import shutil

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
        shutil.copyfile(filename, filename[:-2] + '_best.h5')