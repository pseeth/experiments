import musdb
import os
from networks import *
import torch
from utils import *
import librosa
import argparse
import json
import traceback
from mwf import MWF

parser = argparse.ArgumentParser()
parser.add_argument("--run_directory")
parser.add_argument("--track_id")
parser.add_argument("--estimate_dir")
parser.add_argument("--base_directory")
parser.add_argument("--musdb_directory")
parser.add_argument("--multiple_models", action='store_true')
parser.add_argument("--mwf", action='store_true')
args = parser.parse_args()

mus = musdb.DB(root_dir=args.musdb_directory, is_wav=True)
run_directory = args.run_directory
estimate_directory = os.path.join(args.base_directory, args.estimate_dir, run_directory.split('/')[-1])
os.makedirs(estimate_directory, exist_ok=True)

def load_model(run_directory, device_target):
    if args.checkpoint is None:
        saved_model_path = os.path.join(run_directory, 'model.h5')
    with open(os.path.join(run_directory, 'params.json'), 'r') as f:
        params = json.load(f)

    device = torch.device(device_target)
    if 'baseline' not in run_directory:
        model = DeepAttractor(input_size=int(params['n_fft']/2 + 1),
                       sample_rate=params['sample_rate'],
                       hidden_size=params['hidden_size'], 
                       num_layers=params['num_layers'],
                       dropout=params['dropout'], 
                       num_attractors=params['num_attractors'],
                       embedding_size=params['embedding_size'],
                       activation_type=params['activation_type'],
                       projection_size=params['projection_size'],
                       num_clustering_iterations=params['num_clustering_iterations'],
                       clustering_type=params['clustering_type'],
                       attractor_function_type=params['attractor_function_type'],
                       normalize_embeddings=params['normalize_embeddings'],
                       embedding_activation=params['embedding_activation'],
                       covariance_type=params['covariance_type'],
                       threshold=params['threshold'],
                       use_likelihoods=params['use_likelihoods']).to(device)
    else:
        model = MaskEstimation(input_size=int(params['n_fft']/2 + 1),
                         sample_rate=params['sample_rate'],
                         hidden_size=params['hidden_size'], 
                         num_layers=params['num_layers'],
                         dropout=params['dropout'], 
                         num_sources=params['num_attractors'],
                         projection_size=params['projection_size'],
                         activation_type=params['activation_type']).to(device)

    model.eval()
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['state_dict'])
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

def separate(track):
    print(track.path, track.rate, args.track_id)
    global model
    print(model)
    mixture = track.audio
    labels = ['vocals', 'drums', 'bass', 'other', 'accompaniment']
    estimates = {l: np.zeros(track.audio.shape) for l in labels}
    for channel in range(mixture.shape[-1]):
        mix = mixture[:, channel]
        _, mix = mask_mixture(1, mix, params['n_fft'], params['hop_length'])
        log_spec = whiten(transform(mix, params['n_fft'], params['hop_length']))
        with torch.no_grad():
            input_data = torch.from_numpy(log_spec).unsqueeze(0).requires_grad_().to(device)
            if args.multiple_models:
                masks = []
                for model in models:
                    if 'DeepAttractor' in str(model):
                        one_hot = torch.from_numpy(np.eye(params['num_attractors'], params['num_attractors'])).unsqueeze(0).float().requires_grad_().to(device)
                        masks_ = model(input_data, one_hot)[0]
                    elif 'MaskEstimation' in str(model):
                        masks_ = model(input_data)[0]
                    masks_ = masks_.squeeze(0).cpu().data.numpy()
                    masks.append(masks_[:, :, 0])
                masks = np.stack(masks, axis=-1)
            else:
                if 'DeepAttractor' in str(model):
                    one_hot = torch.from_numpy(np.eye(params['num_attractors'], params['num_attractors'])).unsqueeze(0).float().requires_grad_().to(device)
                    masks = model(input_data, one_hot)[0]
                elif 'MaskEstimation' in str(model):
                    masks = model(input_data)[0]
                masks = masks.squeeze(0).cpu().data.numpy()
        
        
        residual = mix
        
        for i in range(masks.shape[-1]):
            mask = masks[:, :, i].T
            source = mask_mixture(mask, mix, params['n_fft'], params['hop_length'])[0]
            residual -= source
            estimates[labels[i]][:, channel] = source
            if masks.shape[-1] == 4:
                if labels[i] == 'vocals':
                    estimates['accompaniment'][:, channel] = residual

    if args.mwf:
        estimates = {label: estimates[label] for label in labels[:-1]}
        estimates = MWF(track.audio, estimates)
        
    return estimates

def prep_model_to_device(device):
    models = []
    if args.multiple_models:
        targets = ['drums', 'bass', 'other']
        model, params, device = load_model(run_directory, device)
        models.append(model)
        for target in targets:
            model, params, device = load_model(run_directory.replace('vocals', target), device)
            models.append(model)
        return models, params, device
    else:
        model, params, device = load_model(run_directory, device)
        return model, params, device

tracknames = []
for _, folder, _ in os.walk(os.path.join(mus.root_dir, "test")):
    if len(folder) > 0:
        tracknames.append(folder)
tracknames = sorted(tracknames)[0]
track = mus.load_mus_tracks("test", tracknames=[tracknames[int(args.track_id)]])
track_estimate_directory = os.path.join(estimate_directory, "test", tracknames[int(args.track_id)])
perform_separation = False

if os.path.exists(track_estimate_directory):
    separated_tracks = [x for x in os.listdir(track_estimate_directory) if '.wav' in x]
    if len(separated_tracks) < 5:
        perform_separation = True
else:
    perform_separation = True
    
if perform_separation:
    device_target = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model, params, device = prep_model_to_device(device_target)
        if args.multiple_models:
            models = model
        print("Separating with model %s" % run_directory)
        mus.run(separate, tracks=track, estimates_dir=estimate_directory)
    except:
        traceback.print_exc()
        model, params, device = prep_model_to_device('cpu')
        if args.multiple_models:
            models = model
        print("CUDA failed! Using CPU. Separating with model %s" % run_directory)
        mus.run(separate, tracks=track, estimates_dir=estimate_directory)