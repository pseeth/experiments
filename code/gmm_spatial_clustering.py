import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
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


def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)
    # import matplotlib.pyplot as plt
    # if X.shape[-1] > 1:
    #     plt.subplot(121)
    #     plt.scatter(X[::4, 0], X[::4, 1])
    #     plt.subplot(122)
    #     plt.scatter(Y[::4, 0], Y[::4, 1])
    #     plt.show()
    # else:
    #     plt.subplot(121)
    #     plt.hist(X, bins=100)
    #     plt.subplot(122)
    #     plt.hist(Y, bins=100)
    #     plt.show()

    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2

def gmm_spatial_clustering(mix_stft, mix_log_magnitude, sr, num_sources, n_fft, covariance_type='full'):
    fit_weights = np.max(mix_log_magnitude, axis=-1).flatten() > -10
    score_weights = np.max(mix_log_magnitude, axis=-1).flatten() > -40
    interlevel_difference, interphase_difference = extract_spatial_features(mix_stft, n_fft, sr)
    features = np.vstack([np.sin(interphase_difference).flatten(),
                          np.cos(interphase_difference).flatten()]).T
    pca = PCA(n_components=1)

    features = pca.fit_transform(features)
    clusterers = []
    model_scores = []
    aics = []

    for num in range(1, num_sources+1):
        # Find 1 vs 2 clusters. If 1's BIC > 2's BIC, weight this example very low in the 2 speaker condition.
        # Difference in BIC?
        source_share = (1 / float(num))
        clusterer = GaussianMixture(n_components=num,
                                    covariance_type=covariance_type,
                                    weights_init=[source_share for i in range(num)],
                                    init_params='kmeans',
                                    warm_start=True)

        clusterer.fit(features[fit_weights])
        #clusterer.fit(features[score_weights])
        clusterers.append(clusterer)

        model_score = clusterer.score(features[fit_weights])
        model_scores.append(np.exp(model_score))
        aics.append(clusterer.aic(features[fit_weights]))

        assignments = clusterer.predict_proba(features)

        labels = np.zeros(assignments.shape)
        argmax = np.argmax(assignments, axis=-1)
        labels[np.arange(labels.shape[0]), argmax] = 1.0


        cluster_sizes = np.sum(labels[fit_weights], axis=0)
        cluster_sizes /= (np.sum(cluster_sizes) + 1e-6)
        cluster_size_weight = ((source_share - np.abs(cluster_sizes - source_share)) * 2)[0] + 1e-3
        #weight = np.exp(clusterer.score(features[fit_weights])) * cluster_size_weight
        assignments = assignments.reshape(mix_stft.shape[:-1] + (-1,))


    # import matplotlib.pyplot as plt
    # if features.shape[-1] > 1:
    #     plt.scatter(features[fit_weights][:, 0], features[fit_weights][:, 1])
    #     plt.show()
    # else:
    #     plt.hist(features[fit_weights][:, 0], bins=100)
    #     plt.show()
    #Overlap between 1 source model and 2 source model. Higher is better.
    js_divergence = gmm_js(clusterers[0], clusterers[1])
    #print(model_scores, aics, js_divergence, cluster_size_weight)
    weight = max(0, (js_divergence) * cluster_size_weight)

    # likelihood_scores = likelihood_scores.reshape(mix_stft.shape[:-1] + (-1,)) * assignments
    # likelihood_scores /= np.mean(likelihood_scores.reshape((-1, assignments.shape[-1])), axis=0)
    # likelihood_scores = np.max(likelihood_scores, axis=-1)
    # likelihood_scores *= cluster_size_weight
    #
    posterior_scores = (2*np.abs(np.max(assignments, axis=-1) - .5))
    posterior_scores *= weight

    return assignments, posterior_scores + 1e-6