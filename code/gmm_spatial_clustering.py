import numpy as np
from sklearn.mixture import GaussianMixture
import librosa
import utils

def whiten(X, fudge=1E-18):
    Xcov = np.dot(X.T, X)
    d, V = np.linalg.eigh(Xcov)
    D = np.diag(1. / np.sqrt(d + fudge))
    W = np.dot(np.dot(V, D), V.T)
    X_white = np.dot(X, W)
    return X_white

def transform(data, n_fft, hop_length):
    n = len(data)
    data = librosa.util.fix_length(data, n + n_fft // 2)
    stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length).T
    log_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return log_spec, stft

def mask_mixture(source_mask, mix, n_fft, hop_length):
    n = len(mix)
    mix = librosa.util.fix_length(mix, n + n_fft // 2)
    mix_stft = librosa.stft(mix, n_fft=n_fft, hop_length=hop_length)
    masked_mix = mix_stft * source_mask
    source = librosa.istft(masked_mix, hop_length=hop_length, length=n)
    return source

def multichannel_stft(mix, n_fft, hop_length):
    mix_stft = []
    mix_log_magnitude = []
    for ch in range(mix.shape[0]):
        _mix_log_magnitude, _mix_stft = transform(mix[ch], n_fft, hop_length)
        mix_stft.append(_mix_stft)
        mix_log_magnitude.append(_mix_log_magnitude)
    return np.stack(mix_stft, axis=-1), np.stack(mix_log_magnitude, axis=-1)

def extract_spatial_features(mix_stft, n_fft, sr):
    interlevel_difference = np.abs((mix_stft[:, :, 0] + 1e-8) ** 2 / (mix_stft[:, :, 1] + 1e-8) ** 2)
    interlevel_difference = 10 * np.log10(interlevel_difference + 1e-8)

    frequencies = np.expand_dims((2 * np.pi * librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)) / float(sr), axis=0)

    interphase_difference = np.angle(mix_stft[:, :, 1] * np.conj(mix_stft[:, :, 0])) / (frequencies + 1.0)
    return interlevel_difference, interphase_difference

def gmm_spatial_clustering(mix_stft, mix_log_magnitude, sr, num_sources, n_fft, covariance_type='full'):
    fit_weights = np.max(mix_log_magnitude, axis=-1).flatten() > -10
    score_weights = np.max(mix_log_magnitude, axis=-1).flatten() > -40
    interlevel_difference, interphase_difference = extract_spatial_features(mix_stft, n_fft, sr)
    features = np.vstack([np.sin(interphase_difference).flatten(),
                          np.cos(interphase_difference).flatten()]).T

    clusterer = GaussianMixture(n_components=num_sources,
                                covariance_type=covariance_type,
                                weights_init=[.5, .5],
                                init_params='kmeans',
                                warm_start=True)

    clusterer.fit(features[fit_weights])
    clusterer.fit(features[score_weights])

    assignments = clusterer.predict_proba(features)
    cluster_sizes = np.sum(assignments[fit_weights], axis=0)
    cluster_sizes /= np.sum(cluster_sizes)
    cluster_size_weight = ((.5 - np.abs(cluster_sizes - .5)) * 2)[0] + 1e-3

    assignments = assignments.reshape(mix_stft.shape[:-1] + (-1,))
    likelihood_scores = np.exp(clusterer.score_samples(features))
    likelihood_scores /= likelihood_scores.mean()
    likelihood_scores = likelihood_scores.reshape(mix_stft.shape[:-1]) * cluster_size_weight

    posterior_scores = 2*np.abs(np.max(assignments, axis=-1) - .5) * cluster_size_weight

    return assignments, posterior_scores