from sklearn.mixture import GaussianMixture
import librosa
import numpy as np

def transform(data, n_fft, hop_length):
    n = len(data)
    data = librosa.util.fix_length(data, n + n_fft // 2)
    stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length).T
    log_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    return log_spec, stft

def mask_mixture(source_mask, mix):
    n = len(mix)
    mix = librosa.util.fix_length(mix, n + n_fft // 2)
    mix_stft = librosa.stft(mix, n_fft=n_fft, hop_length=hop_length)
    masked_mix = mix_stft * source_mask
    source = librosa.istft(masked_mix, hop_length=hop_length, length=n)
    return source

def multichannel_stft(mix, n_fft, hop_length):
    mix_stft = []
    for ch in range(mix.shape[0]):
        _mix_log_magnitude, _mix_stft = transform(mix[ch], n_fft, hop_length)
        mix_stft.append(_mix_stft)
    return np.stack(mix_stft, axis=-1)

def extract_spatial_features(mix_stft):
    interlevel_difference = np.abs((mix_stft[:, :, 0] + 1e-8) ** 2 / (mix_stft[:, :, 1] + 1e-8)) ** 2
    interlevel_difference = 10 * np.log10(interlevel_difference + 1e-8)

    frequencies = np.expand_dims((2 * np.pi * librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)) / float(sr), axis=0)
    interphase_difference = np.angle(mix_stft[:, :, 0] * np.conj(mix_stft[:, :, 1])) / (frequencies + 1.0)
    return interlevel_difference, interphase_difference

def gmm_spatial_clustering(mix, num_sources, n_fft, hop_length, verbose=True):
    mix_stft = multichannel_stft(mix, n_fft, hop_length)

    weights = np.mean(20 * np.log10(np.abs(mix_stft) + 1e-8), axis=-1).flatten()
    weights -= weights.min()
    weights /= weights.max()
    weights = weights > .5

    interlevel_difference, interphase_difference = extract_spatial_features(mix_stft)
    features = np.vstack([np.sin(interphase_difference).flatten(),
                          np.cos(interphase_difference).flatten()]).T
    features = (features.T * weights).T

    data = features
    clusterer = GaussianMixture(n_components=2, covariance_type='full', init_params='kmeans')
    clusterer.fit(data)
    assignments = clusterer.predict_proba(data)
    assignments = assignments.reshape(mix_stft.shape[:-1] + (-1,))
    scores = np.exp(clusterer.score_samples(data).reshape(mix_stft.shape[:-1]))
    sources = []
    for i in range(assignments.shape[-1]):
        mask = assignments[:, :, i]
        source = np.vstack([mask_mixture(mask.T, mix[i]) for i in range(mix.shape[0])])
        sources.append(source)

    return sources, clusterer.bic(data), scores