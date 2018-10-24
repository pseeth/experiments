import numpy as np
from sklearn.mixture import GaussianMixture
import librosa

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

def gmm_spatial_clustering(mix, sr, num_sources, n_fft, hop_length, covariance_type='full'):
    mix_stft, mix_log_magnitude = multichannel_stft(mix, n_fft, hop_length)
    weights = np.max(mix_log_magnitude, axis=-1).flatten() > -50

    interlevel_difference, interphase_difference = extract_spatial_features(mix_stft, n_fft, sr)
    features = np.vstack([np.sin(interphase_difference).flatten(),
                          np.cos(interphase_difference).flatten()]).T
    features_fit = features[weights]

    clusterer = GaussianMixture(n_components=num_sources,
                                covariance_type=covariance_type,
                                weights_init=[.5, .5])
    clusterer.fit(features_fit)
    assignments = clusterer.predict_proba(features)
    assignments = assignments.reshape(mix_stft.shape[:-1] + (-1,))
    scores = np.exp(clusterer.score_samples(features)) * weights
    scores = scores.reshape(mix_stft.shape[:-1])

    assignments = (assignments == assignments.max(axis=-1)) * weights.reshape(mix_stft.shape[:-1])

    return assignments, clusterer.bic(features), scores