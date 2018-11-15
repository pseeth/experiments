import torch
from torch import nn
from torch.utils.data import DataLoader, sampler, ConcatDataset
from networks import DeepAttractor, MaskEstimation
import utils
from loss import *
from dataset import ScaperLoader
from wsj_dataset import WSJ0
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
parser.add_argument("--log_dir", default=None)
parser.add_argument("--training_folder", default='/mm1/seetharaman/generated/speech/two_speaker/training')
parser.add_argument("--validation_folder", default='/mm1/seetharaman/generated/speech/two_speaker/validation')
parser.add_argument("--dataset_type", default='scaper')
parser.add_argument("--n_clusters", default=2)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--disable-training-stats", action='store_true')
parser.add_argument("--loss_function", default='l1')
parser.add_argument("--target_type", default='psa')
parser.add_argument("--projection_size", default=0)
parser.add_argument("--activation_type", default='sigmoid')
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
parser.add_argument("--clustering_type", default='kmeans')
parser.add_argument("--unfold_iterations", action='store_true')
parser.add_argument("--covariance_type", choices=['spherical', 'diag', 'tied_spherical', 'tied_diag'], default='diag')
parser.add_argument("--covariance_min", default=.5)
parser.add_argument("--fix_covariance", action='store_true')
parser.add_argument("--num_gaussians_per_source", default=1)
parser.add_argument("--weight_method", default='magnitude')
parser.add_argument("--create_cache", action='store_true')
parser.add_argument("--generate_plots", action='store_true')

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
    'dataset_type': args.dataset_type,
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
    'curriculum_learning': args.curriculum_learning,
    'weight_method': args.weight_method,
    'create_cache': args.create_cache
}
