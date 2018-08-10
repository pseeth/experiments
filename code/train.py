import torch
from torch import nn
from torch.utils.data import DataLoader, sampler, ConcatDataset
from networks import DeepAttractor, MaskEstimation
import utils
import loss
from dataset import ScaperLoader
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import warnings
import os
import json
from validate import validate
import subprocess
import time
import pprint

pp = pprint.PrettyPrinter(indent=4)
torch.manual_seed(0)
warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.monitor_interval = 0

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='model')
parser.add_argument("--log_dir", default=None)
parser.add_argument("--training_folder", default='/mm1/seetharaman/generated/speech/two_speaker/training')
parser.add_argument("--validation_folder", default='/mm1/seetharaman/generated/speech/two_speaker/validation')
parser.add_argument("--n_clusters", default=2)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--disable-training-stats", action='store_true')
parser.add_argument("--loss_function", default='l1')
parser.add_argument("--target_type", default='psa')
parser.add_argument("--projection_size", default=0)
parser.add_argument("--activation_type", default='sigmoid')
parser.add_argument("--attractor_loss_function", default='none')
parser.add_argument("--attractor_loss_weight", default=.001)
parser.add_argument("--attractor_function_type", default='ae')
parser.add_argument("--num_clustering_iterations", default=5)
parser.add_argument("--n_fft", default=256)
parser.add_argument("--hop_length", default=64)
parser.add_argument("--hidden_size", default=600)
parser.add_argument("--num_layers", default=4)
parser.add_argument("--dropout", default=.3)
parser.add_argument("--embedding_size", default=40)
parser.add_argument("--learning_rate", default=1e-3)
parser.add_argument("--batch_size", default=40)
parser.add_argument("--initial_length", default=1.0)
parser.add_argument("--curriculum_learning", action='store_true')
parser.add_argument("--num_epochs", default=75)
parser.add_argument("--optimizer", default='adam')
parser.add_argument("--group_sources", default='')
parser.add_argument("--ignore_sources", default='')
parser.add_argument("--source_labels", default='')
parser.add_argument("--baseline", action='store_true')
parser.add_argument("--weight_decay", default=0.0)
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--resume", action='store_true')
parser.add_argument("--normalize_embeddings", action='store_true')
parser.add_argument("--threshold", default=None)
parser.add_argument("--num_workers", default=10)
parser.add_argument("--sample_strategy", default='sequential')
parser.add_argument("--embedding_activation", default='none')
parser.add_argument("--use_enhancement", action='store_true')
parser.add_argument("--reorder_sources", action='store_true')
parser.add_argument("--clustering_type", default='kmeans')
parser.add_argument("--unfold_iterations", action='store_true')
parser.add_argument("--covariance_type", choices=['spherical', 'diag', 'tied_spherical', 'tied_diag'], default='diag')
parser.add_argument("--covariance_min", default=.5)
parser.add_argument("--fix_covariance", action='store_true')
parser.add_argument("--num_gaussians_per_source", default=1)
parser.add_argument("--use_likelihoods", action='store_true')

args = parser.parse_args()

params = {
    'n_fft': int(args.n_fft),
    'input_size': int(int(args.n_fft)/2 + 1),
    'hop_length': int(args.hop_length),
    'hidden_size': int(args.hidden_size),
    'num_layers': int(args.num_layers),
    'dropout': float(args.dropout),
    'embedding_size': int(args.embedding_size),
    'projection_size': int(args.projection_size),
    'activation_type': args.activation_type,
    'embedding_activation': args.embedding_activation,
    'n_clusters': int(args.n_clusters),
    'learning_rate': float(args.learning_rate),
    'batch_size': int(args.batch_size),
    'compute_training_stats': args.disable_training_stats,
    'target_type': args.target_type,
    'training_folder': args.training_folder.split(':'),
    'validation_folder': args.validation_folder,
    'checkpoint': args.checkpoint,
    'loss_function': args.loss_function,
    'attractor_loss_function': args.attractor_loss_function,
    'attractor_alpha': float(args.attractor_loss_weight),
    'attractor_function_type': args.attractor_function_type,
    'num_clustering_iterations': int(args.num_clustering_iterations),
    'use_enhancement': args.use_enhancement,
    'optimizer': args.optimizer,
    'num_epochs': int(args.num_epochs),
    'group_sources': args.group_sources.split('_') if len(args.group_sources) > 0 else [],
    'ignore_sources': args.ignore_sources.split('_') if len(args.ignore_sources) > 0 else [],
    'source_labels': args.source_labels.split('_') if len(args.source_labels) > 0 else [],
    'normalize_embeddings': args.normalize_embeddings,
    'initial_length': float(args.initial_length),
    'threshold': float(args.threshold) if args.threshold is not None else None,
    'num_workers': int(args.num_workers),
    'sample_rate': None,
    'training_stats': None,
    'weight_type': None,
    'lr_decay': .5,
    'l2_reg': float(args.weight_decay),
    'beta1': .9,
    'beta2': .999,
    'patience': 5,
    'save_checkpoints': True,
    'validate': True,
    'grad_clip': 100,
    'clustering_type': args.clustering_type,
    'covariance_type': args.covariance_type,
    'covariance_min': float(args.covariance_min),
    'fix_covariance': args.fix_covariance,
    'num_gaussians_per_source': int(args.num_gaussians_per_source),
    'use_likelihoods': args.use_likelihoods,
    'curriculum_learning': args.curriculum_learning
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if os.path.isdir(os.path.join(args.log_dir, 'checkpoints')):
    if args.overwrite:
        print('Deleting existing directory: ', args.log_dir)
        subprocess.call(['rm','-rf', args.log_dir])
        print("Found existing directory. Sleeping for 5 seconds after deleting existing directory!")
        time.sleep(5)

writer = SummaryWriter(log_dir=args.log_dir)
args.log_dir = writer.file_writer.get_logdir()

os.makedirs(os.path.join(args.log_dir, 'checkpoints'), exist_ok=True)

dataset = []
for i in range(len(params['training_folder'])):
    dataset.append(ScaperLoader(folder=params['training_folder'][i], 
                           length=params['initial_length'], 
                           n_fft=params['n_fft'], 
                           hop_length=params['hop_length'], 
                           output_type=params['target_type'],
                           group_sources=params['group_sources'],
                           ignore_sources=params['ignore_sources'],
                           source_labels=params['source_labels']))

dataset = ConcatDataset(dataset) if len(dataset) > 1 else dataset[0]

val_dataset = ScaperLoader(folder=params['validation_folder'], 
                           length=params['initial_length'], 
                           n_fft=params['n_fft'], 
                           hop_length=params['hop_length'], 
                           output_type=params['target_type'],
                           group_sources=params['group_sources'],
                           ignore_sources=params['ignore_sources'],
                           source_labels=params['source_labels'])

if args.sample_strategy == 'sequential':
    sample_strategy = sampler.SequentialSampler(dataset)
elif args.sample_strategy == 'random':
    sample_strategy = sampler.RandomSampler(dataset)
    
dataloader = DataLoader(dataset, batch_size=params['batch_size'], num_workers=params['num_workers'], sampler=sample_strategy)

dummy_input, _, _, _, dummy_one_hot = dataset[0]
params['num_attractors'] = dummy_one_hot.shape[-1]
params['num_sources'] = params['num_attractors']
params['sample_rate'] = dataset.sr
dataset.reorder_sources = args.reorder_sources
val_dataset.reorder_sources = args.reorder_sources

pp.pprint(params)

class_func = MaskEstimation if args.baseline else DeepAttractor
model = utils.load_class_from_params(params, class_func).to(device)

if not params['compute_training_stats']:
    mean, std = utils.compute_statistics_on_dataset(dataloader, device)
    params['training_stats'] = {'mean': mean, 'std': std + 1e-7}
    dataset.stats = params['training_stats']
    val_dataset.stats = params['training_stats']

dataset.whiten_data = True
val_dataset.whiten_data = True

num_iterations = len(dataloader)
parameters = filter(lambda p: p.requires_grad, model.parameters())

if params['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(parameters,
                                 lr=params['learning_rate'],
                                 betas=(params['beta1'], params['beta2']),
                                 weight_decay=params['l2_reg'])
elif params['optimizer'] == 'rmsprop':
    optimizer = torch.optim.RMSprop(parameters,
                                   lr=params['learning_rate'],
                                   weight_decay=params['l2_reg'])
elif params['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(parameters,
                                   lr=params['learning_rate'],
                                   weight_decay=params['l2_reg'])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=params['lr_decay'], patience=params['patience'])
checkpoints = sorted(os.listdir(os.path.join(args.log_dir, 'checkpoints')))

if args.resume and len(checkpoints) > 0:
    checkpoint_path = os.path.join(args.log_dir, 'checkpoints/latest.h5')
    checkpoint = torch.load(checkpoint_path)
    if checkpoint['epoch'] == params['num_epochs']:
        raise ValueError("Run completed! Exiting.")
    else:
        #time_since_last_checkpoint = (time.time() - os.path.getmtime(checkpoint_path)) / 3600
        #if time_since_last_checkpoint < .5:
        #    raise ValueError("Job is still running", args.log_dir)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        utils.move_optimizer(optimizer, device)
        epochs = trange(checkpoint['epoch'], params['num_epochs'])
        n_iter = num_iterations*checkpoint['epoch']
        print("Resuming job: ", args.log_dir)
else:
    epochs = trange(params['num_epochs'])
    n_iter = 0
    with open(os.path.join(args.log_dir, 'params.json'), 'w') as f:
        json.dump(params, f, sort_keys=True, indent=4)

if params['loss_function'] == 'l1':
    loss_function = nn.L1Loss()
elif params['loss_function'] == 'mse':
    loss_function = nn.MSELoss()
elif params['loss_function'] == 'kl':
    loss_function = nn.KLDivLoss()
elif params['loss_function'] == 'weighted_l1':
    l1_loss = nn.L1Loss()
    loss_function = loss.WeightedL1Loss(loss_function=l1_loss)

if params['attractor_loss_function'] == 'none':
    attractor_loss_function = None
elif params['attractor_loss_function'] == 'sparse':
    attractor_loss_function = loss.sparse_orthogonal_loss
    attractor_loss_weights = (1., 0.)
elif params['attractor_loss_function'] == 'orth':
    attractor_loss_function = loss.sparse_orthogonal_loss
    attractor_loss_weights = (0., 1.)
elif params['attractor_loss_function'] == 'sparse_orth':
    attractor_loss_function = loss.sparse_orthogonal_loss
    attractor_loss_weights = (1., 1.)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
module = model.module if torch.cuda.device_count() > 1 else model

utils.show_model(model)

val_losses = []

for epoch in epochs:
    if args.unfold_iterations:
        if (epoch % int(params['num_epochs'] / 5) == 0):
            n_iterations = int(params['num_epoch'] / epoch)
            module.clusterer.n_iterations = min(params['num_clustering_iterations'], n_iterations)

    if args.curriculum_learning:
        if epoch >= int(params['num_epochs'] / 5):
            # Lengthen sequences for learning
            dataset.length = 1.0
            val_dataset.length = 1.0
                
    progress_bar = trange(num_iterations)
    epoch_loss = []
    for (spectrogram, magnitude_spectrogram, source_spectrograms, source_ibms, one_hots) in dataloader:
        spectrogram = spectrogram.to(device).requires_grad_()
        magnitude_spectrogram = magnitude_spectrogram.to(device).unsqueeze(-1).requires_grad_()
        source_spectrograms = source_spectrograms.float().to(device)
        source_ibms = source_ibms.to(device).float()
        one_hots = one_hots.float().to(device).requires_grad_()
        
        optimizer.zero_grad()
        source_masks, attractors, embedding, log_likelihoods = model(spectrogram, one_hots)
        
        source_estimates = source_masks * magnitude_spectrogram
        loss = loss_function(source_estimates, source_spectrograms)
        writer.add_scalar('mask_loss/scalar', loss.item(), n_iter)

        if not args.baseline:
            if attractor_loss_function is not None:
                if isinstance(attractors, (tuple,)):
                    attractor_data = attractors[1]
                else:
                    attractor_data = attractors
                attractor_loss = attractor_loss_function(attractor_data, weights=attractor_loss_weights)
                writer.add_scalar('attr_loss/scalar', attractor_loss.item(), n_iter)
                loss += params['attractor_alpha']*attractor_loss
        
            writer.add_scalar('inv_variance/scalar', 1/attractors[1].mean().item(), n_iter)
            writer.add_scalar('log_likelihood/scalar', log_likelihoods.mean().item(), n_iter)
            writer.add_scalar('embedding/scalar', embedding.norm(p=2).item(), n_iter)

        if np.isnan(loss.item()):
            print('Loss went to nan - deleting existing directory: ', args.log_dir)
            subprocess.call(['rm','-rf', args.log_dir])
            raise ValueError("Loss went to nan - aborting training.")
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
        optimizer.step()

        progress_bar.set_description(str(loss.item()))
        progress_bar.update(1)
        writer.add_scalar('iter_loss/scalar', loss.item(), n_iter)
        epoch_loss.append(loss.item())
        n_iter += 1
        
    epoch_loss = np.mean(epoch_loss)
    writer.add_scalar('epoch_loss/scalar', epoch_loss, epoch)

    is_best = True
    if params['validate']:
        val_loss = validate(module, val_dataset, writer, n_iter, params, device, loss_function)
        writer.add_scalar('val_loss/scalar', val_loss, epoch)
        val_losses.append(val_loss)
        is_best = (val_loss == min(val_losses))
        scheduler.step(val_loss)

    if params['save_checkpoints']:
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join(args.log_dir, 'checkpoints', 'latest.h5'))