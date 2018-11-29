import torch
from torch import nn
from tqdm import trange, tqdm
import multiprocessing
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import networks
from enums import *
import os

OutputTargetMap = {
    'estimates': ['sources'],
    'embedding': ['assignments', 'weights']
}

class Trainer():
    def __init__(self,
                 output_folder,
                 train_data,
                 validation_data,
                 model,
                 options):

        self.prepare_directories(output_folder)
        self.model = self.build_model(model)
        self.device = (torch.device('cuda') if options['device'] == 'cuda'
            else torch.device('cpu'))
        self.model = self.model.to(self.device)

        self.writer = SummaryWriter(log_dir=self.output_folder)
        self.loss_dictionary = {target: (LossFunctions[fn.upper()].value(), float(weight))
                            for (fn, target, weight) in options['loss_function']}
        self.loss_keys = sorted(list(self.loss_dictionary))
        self.options = options

        self.dataloaders = {
            'training': self.create_dataloader(train_data),
            'validation': self.create_dataloader(validation_data)
        }

        self.optimizer, self.scheduler = self.create_optimizer_and_scheduler(self.model, self.options)
        self.module = self.model
        if options['data_parallel'] and options['device'] == 'cuda':
            self.model = nn.DataParallel(model)
            self.module = self.model.module
        self.model.train()

    @staticmethod
    def build_model(model):
        if type(model) is str:
            if '.json' in model:
                model = networks.SeparationModel(model)
            else:
                model_dict = torch.load(model)
                model = model_dict['model']
                model.load_state_dict(model_dict['state_dict'])
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
        loss = []
        for key in self.loss_keys:
            loss_function = self.loss_dictionary[key][0]
            weight = self.loss_dictionary[key][1]
            target_keys = OutputTargetMap[key]
            arguments = [outputs[key]] + [targets[t] for t in target_keys]
            _loss = weight * loss_function(*arguments)
            self.check_loss(_loss)
            loss.append(_loss)
        return loss

    def check_loss(self, loss):
        if np.isnan(loss.item()):
            raise ValueError("Loss went to nan - aborting training.")

    def resume(self):
        return

    def fit_epoch(self, dataloader):
        for data in dataloader:
            for key in data:
                data[key] = data[key].float().to(self.device)
                if key in self.input_keys:
                    data[key] = data[key.requires_grad_()]
            

        return

    def fit(self):
        return

    def validate(self):
        return

    def transform(self):
        return
