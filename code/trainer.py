import torch
from torch import nn
from tqdm import trange, tqdm
import multiprocessing
from tensorboardX import SummaryWriter
from enum import Enum
from torch.utils.data import sampler, DataLoader
from loss import DeepClusteringLoss, PermutationInvariantLoss
import numpy as np
import networks

class Samplers(Enum):
    SEQUENTIAL = sampler.SequentialSampler
    RANDOM = sampler.RandomSampler

class LossFunctions(Enum):
    DPCL = DeepClusteringLoss
    PIT = PermutationInvariantLoss
    L1 = nn.L1Loss
    MSE = nn.MSELoss
    KL = nn.KLDivLoss

class Optimizers(Enum):
    ADAM = torch.optim.Adam
    RMSPROP = torch.optim.RMSprop
    SGD = torch.optim.SGD

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
                 loss_tuples,
                 options=None):

        defaults = {
            'num_epochs': 100,
            'learning_rate': 1e-3,
            'learning_rate_decay': .5,
            'patience': 5,
            'batch_size': 40,
            'num_workers': multiprocessing.cpu_count(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'sample_strategy': 'sequential',
            'data_parallel': torch.cuda.device_count() > 1,
            'weight_decay': 0.0,
            'optimizer': 'adam'
        }

        options = {**defaults, **(options if options else {})}

        if type(model) is str:
            if '.json' in model:
                model = networks.SeparationModel(model)
            else:
                model_dict = torch.load(model)
                model = model_dict['model']
                model.load_state_dict(model_dict['state_dict'])
        self.model = model
        self.device = (torch.device('cuda') if options['device'] == 'cuda'
            else torch.device('cpu'))
        self.model = self.model.to(self.device)
        self.module = self.model
        if options['data_parallel'] and options['device'] == 'cuda':
            self.model = nn.DataParallel(model)
            self.module = self.model.module
        self.model.train()

        self.writer = SummaryWriter(log_dir=output_folder)
        self.loss_dictionary = {target: (LossFunctions[fn.upper()].value(), weight)
                            for (fn, target, weight) in loss_tuples}
        self.loss_keys = sorted(list(self.loss_dictionary))
        self.options = options

        #self.dataloaders = {
        #    'training': self.create_dataloader(train_data),
        #    'validation': self.create_dataloader(validation_data)
        #}

        self.optimizer = self.create_optimizer(self.model, self.options)

    def create_dataloader(self, dataset):
        _sampler = Samplers[self.options['sample_strategy'].upper()].value(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=self.options['batch_size'],
                                num_workers=self.options['num_workers'],
                                sampler=_sampler)
        return dataloader

    def create_optimizer(self, model, options):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Optimizers[options['optimizer'].upper()].value(
                                parameters,
                                lr=options['learning_rate'],
                                weight_decay=options['weight_decay'])
        return optimizer

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
            loss.append(weight * loss_function(*arguments))
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