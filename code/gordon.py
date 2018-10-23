import torch
import numpy as np

# Make some random data to test deep clustering
batch_size=5
num_points=30
embedding_size=8
embeddings = torch.rand(batch_size, num_points, embedding_size)
weights = torch.rand(batch_size, num_points)
a0 = torch.randint(0,2, (batch_size, num_points))
assignments = torch.stack([a0, 1-a0], dim=-1)

def classic_deep_clustering(embeddings, assignments, weights):
    batch_size, num_points, embedding_size = embeddings.shape
    weights = weights.view(batch_size, num_points, 1)
    embeddings = weights.expand_as(embeddings) * embeddings
    assignments = weights.expand_as(assignments) * assignments
    
    # -- get normalization factor (normalize by sum over all applied weights)
    # See (7) from ALTERNATIVE OBJECTIVE FUNCTIONS FOR DEEP CLUSTERING, Wang et al, 2017
    # for how to apply weights.  Innermost square() because we're working with sqrt weights
    # Then use (\sum_{i} w_{ii})^2 = \sum_{ij} w_i * w_j and sum over batches
    count = torch.sum(torch.sum(weights**2, dim=1)**2)
    
    embeddings_transpose = embeddings.transpose(2, 1)
    assignments_transpose = assignments.transpose(2, 1)
    
    vTv = torch.matmul(embeddings_transpose, embeddings)
    vTy = torch.matmul(assignments_transpose, assignments)
    yTy = torch.matmul(assignments_transpose, assignments)
    return (torch.sum(vTv**2) - 2*torch.sum(vTy**2) + torch.sum(yTy**2)) / count 

L = classic_deep_clustering(embeddings, assignments, weights)


# These functions should move to the dataloader
def get_mag_weight_Rsqrt(mag):
    """ Compute magnitude ratio weights from ALTERNATIVE OBJECTIVE FUNCTIONS
        FOR DEEP CLUSTERING, Wang et al, ICASSP 2017.
        Input:
            mag: (n_frames, n_bins) magnitude spectrogram of mixture
        output:
            weights: (n_frames, n_bins) for use in deep clustering objective """
    return np.sqrt((mag / np.sum(mag)) * mag.shape[0] * mag.shape[1])

def get_mag_weight_B(source_magnitudes, thresh=-40.):
    """ Compute voice activity weights from ALTERNATIVE OBJECTIVE FUNCTIONS
        FOR DEEP CLUSTERING, Wang et al, ICASSP 2017.
        Input:
            source_magnitudes: (n_frames, n_bins, n_sources) magnitude
                               spectrograms of isolated sources
                               
            thresh:            distance from max dB in input for weight of zero
        output:
            weights: (n_frames, n_bins) for use in deep clustering objective """
    logmag = 20 * np.log10(source_magnitudes + 1e-20)
    is_above = (logmag - np.max(logmag)) > thresh
    return np.max(is_above, axis=2).astype(np.float32)
