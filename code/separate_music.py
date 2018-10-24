import os
from networks import *
from utils import *
import librosa
import json
import argparse
from scipy.io import wavfile
import numpy as np

def load_model(run_directory):
    saved_model_path = os.path.join(run_directory, 'checkpoints/latest_best.h5')
    with open(os.path.join(run_directory, 'params.json'), 'r') as f:
        params = json.load(f)

    device = torch.device('cuda', 1)
    class_func = MaskEstimation if 'baseline' in run_directory else DeepAttractor
    model = load_class_from_params(params, class_func).to(device)

    model.eval()
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    show_model(model)
    return model, params, device

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


def separate(mixture, model, params, device):
    labels = ['vocals', 'drums', 'bass', 'other']
    estimates = {l: np.zeros(mixture.shape) for l in labels}
    for channel in range(mixture.shape[-1]):
        mix = mixture[:, channel]
        _, mix = mask_mixture(1, mix, params['n_fft'], params['hop_length'])
        log_spec = whiten(transform(mix, params['n_fft'], params['hop_length']))
        with torch.no_grad():
            input_data = torch.from_numpy(log_spec).unsqueeze(0).requires_grad_().to(device)
            if 'DeepAttractor' in str(model):
                one_hot = torch.from_numpy(np.eye(params['num_attractors'], params['num_attractors'])).unsqueeze(
                    0).float().requires_grad_().to(device)
                masks = model(input_data, one_hot)[0]
            elif 'MaskEstimation' in str(model):
                masks = model(input_data)[0]
            masks = masks.squeeze(0).cpu().data.numpy()

        for i in range(masks.shape[-1]):
            mask = masks[:, :, i].T
            source, _ = mask_mixture(mask, mix, params['n_fft'], params['hop_length'])
            estimates[labels[i]][:, channel] = source

    return estimates

def save_estimates(estimates, target_directory, orig_sr=48000, target_sr=32000):
    os.makedirs(os.path.join(target_directory, 'estimates'), exist_ok=True)
    maxv = np.iinfo(np.int16).max
    for label in estimates:
        estimates[label] = estimates[label].T
        save_path = os.path.join(target_directory, 'estimates', label + '.wav')
        estimates[label] = librosa.resample(estimates[label], orig_sr, target_sr, res_type='kaiser_fast')
        estimates[label] = (estimates[label] * maxv).astype(np.int16)
        wavfile.write(save_path, target_sr, estimates[label].T)

def save_references(mixture_folder, target_directory, orig_sr=48000, target_sr=32000):
    os.makedirs(os.path.join(target_directory, 'references'), exist_ok=True)
    maxv = np.iinfo(np.int16).max
    for label in ['vocals', 'bass', 'drums', 'other']:
        file_path = os.path.join(mixture_folder, label + '.wav')
        save_path = os.path.join(target_directory, 'references', label + '.wav')
        source, sr = librosa.load(file_path,
                                   sr=orig_sr,
                                   mono=False)
        source = librosa.resample(source, orig_sr, target_sr, res_type='kaiser_fast')
        source = (source*maxv).astype(np.int16)
        wavfile.write(save_path, target_sr, source.T)

def run(run_directory, mixture_folder):
    file_name = mixture_folder.split('/')[-1]
    print(file_name)
    target_directory = os.path.join(run_directory, 'output', file_name)

    model, params, device = load_model(run_directory)
    mixture, sr = librosa.load(os.path.join(mixture_folder, 'mixture.wav'),
                               sr=params['sample_rate'],
                               mono=False)
    estimates = separate(mixture.T, model, params, device)

    save_estimates(estimates,
                   target_directory,
                   orig_sr=params['sample_rate'],
                   target_sr=48000)

    save_references(mixture_folder,
                    target_directory,
                    orig_sr=params['sample_rate'],
                    target_sr=48000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_directory")
    parser.add_argument("--mixture_folder")
    args = parser.parse_args()

    run(args.run_directory, args.mixture_folder)




