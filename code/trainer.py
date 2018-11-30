import torch
from torch import nn
from tqdm import trange, tqdm
import multiprocessing
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from nussl import networks
from enums import *
import os
import shutil

OutputTargetMap = {
    'estimates': ['source_spectrograms'],
    'embedding': ['assignments', 'weights']
}

class Trainer():
    def __init__(self,
                 output_folder,
                 train_data,
                 validation_data,
                 model,
                 options,
                 verbose=True):

        self.verbose = verbose
        self.prepare_directories(output_folder)
        self.model = self.build_model(model)
        self.device = torch.device("cuda" if options['device'] == 'cuda' else "cpu")
        self.model = self.model.to(self.device)

        self.writer = SummaryWriter(log_dir=self.output_folder) if verbose else None
        self.loss_dictionary = {target: (LossFunctions[fn.upper()].value(), float(weight))
                            for (fn, target, weight) in options['loss_function']}
        self.loss_keys = sorted(list(self.loss_dictionary))
        self.options = options
        self.num_epoch = 0

        self.dataloaders = {
            'training': self.create_dataloader(train_data),
            'validation': self.create_dataloader(validation_data)
        }

        self.optimizer, self.scheduler = self.create_optimizer_and_scheduler(self.model, self.options)
        self.module = self.model
        if options['data_parallel'] and options['device'] == 'cuda':
            self.model = nn.DataParallel(self.model)
            self.module = self.model.module
        self.model.train()

    @staticmethod
    def build_model(model):
        if type(model) is str:
            if '.json' in model:
                model = networks.SeparationModel(model)                
        if type(model) is dict:
            model = networks.SeparationModel(model)
        return model

    def prepare_directories(self, output_folder):
        self.output_folder = output_folder
        self.checkpoint_folder = os.path.join(output_folder, 'checkpoints')
        self.config_folder = os.path.join(output_folder, 'config')

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        os.makedirs(self.config_folder, exist_ok=True)

    def create_dataloader(self, dataset):
        if not dataset:
            return None
        _sampler = Samplers[self.options['sample_strategy'].upper()].value(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=self.options['batch_size'],
                                num_workers=self.options['num_workers'],
                                sampler=_sampler)
        return dataloader

    def create_optimizer_and_scheduler(self, model, options):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Optimizers[options['optimizer'].upper()].value(
                                parameters,
                                lr=options['learning_rate'],
                                weight_decay=options['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               'min', 
                                                               factor=options['learning_rate_decay'],
                                                               patience=options['patience'])
        return optimizer, scheduler

    def calculate_loss(self, outputs, targets):
        if self.module.layers['mel_projection'].num_mels > 0:
            if 'assignments' in targets:
                targets['assignments'] = self.module.project_assignments(targets['assignments'])
            if 'weights' in targets:
                targets['weights'] = self.module.project_assignments(targets['weights'])
        loss = {}
        for key in self.loss_keys:
            loss_function = self.loss_dictionary[key][0]
            weight = self.loss_dictionary[key][1]
            target_keys = OutputTargetMap[key]
            arguments = [outputs[key]] + [targets[t] for t in target_keys]
            _loss = weight * loss_function(*arguments)
            self.check_loss(_loss)
            loss[key] = _loss
        return loss

    def check_loss(self, loss):
        if np.isnan(loss.item()):
            raise ValueError("Loss went to nan - aborting training.")

    def run_epoch(self, dataloader):
        epoch_loss = 0
        step = 0
        num_batches = len(dataloader)
        for data in dataloader:
            for key in data:
                data[key] = data[key].float().to(self.device)
                if key not in self.loss_keys and self.model.training:
                    data[key] = data[key].requires_grad_()
            output = self.model(data)
            loss = self.calculate_loss(output, data)
            loss['total_loss'] = sum(list(loss.values()))
            epoch_loss += loss['total_loss'].item()

            if self.model.training:
                self.log_to_tensorboard({k: loss[k].item() for k in loss}, step + self.num_epoch * num_batches, 'iter')
                self.optimizer.zero_grad()
                loss['total_loss'].backward()
                self.optimizer.step()
            step += 1
        return {'loss': epoch_loss / float(num_batches)}

    def log_to_tensorboard(self, data, step, scope):
        if self.verbose:
            prefix = 'training' if self.model.training else 'validation'
            for key in data:
                label = os.path.join(prefix, key)
                self.writer.add_scalar(label, data[key], step)

    def fit(self):
        progress_bar = trange(self.num_epoch, self.options['num_epochs'])
        lowest_validation_loss = np.inf
        for self.num_epoch in progress_bar:
            epoch_loss = self.run_epoch(self.dataloaders['training'])
            self.log_to_tensorboard(epoch_loss, self.num_epoch, 'epoch')
            validation_loss = self.validate(self.dataloaders['validation'])
            self.save(validation_loss <= lowest_validation_loss)

            progress_bar.update(1)
            progress_bar.set_description('Loss: {:.4f}'.format(epoch_loss['loss']))

            if self.num_epoch == 0:
                for key in self.dataloaders:
                    self.dataloaders[key].dataset.toggle_cache()

    def validate(self, dataloader):
        if not dataloader:
            return 0
        self.model.eval()
        with torch.no_grad():
            validation_loss = self.run_epoch(dataloader)
        self.log_to_tensorboard(validation_loss, self.num_epoch, 'epoch')
        self.model.train()
        self.scheduler.step(validation_loss['loss'])
        return validation_loss['loss']
    
    def save(self, best):
        prefix = 'best' if best else 'latest'
        optimizer_path = os.path.join(self.checkpoint_folder, f'{prefix}.opt.pth')
        model_path = os.path.join(self.checkpoint_folder, f'{prefix}.model.pth')
        dataset_options = self.dataloaders['training'].dataset['options']
        metadata = {
            'n_fft': dataset_options['n_fft'],
            'hop_length': dataset_options['hop_length'],
            'format': dataset_options['format']
        }
        
        optimizer_state = {
            'optimizer': self.optimizer.state_dict(),
            'num_epoch': self.num_epoch
        }

        torch.save(optimizer_state, optimizer_path)
        self.module.save(model_path, {'metadata': metadata})
    
    def resume(self, prefix='best'):
        optimizer_path = os.path.join(self.checkpoint_folder, f'{prefix}.opt.pth')
        model_path = os.path.join(self.checkpoint_folder, f'{prefix}.model.pth')

        optimizer_state = torch.load(optimizer_path)

        model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = model_dict['model']
        model.load_state_dict(model_dict['state_dict'])

        self.model = self.load_model(model_path)
        return