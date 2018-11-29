import multiprocessing
import os
from config.utils.defaults import load_from_json
from trainer import Trainer
import argparse
from enums import *
from torch.utils.data import ConcatDataset

def parse():
    parser = argparse.ArgumentParser(
        description='Parse {model, dataset, train} JSONs',
    )
    parser.add_argument(
        '--train',
        required=True,
        help='Path to JSON containing training configuration',
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to JSON containing model configuration',
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to JSON containing dataset configuration',
    )
    parser.add_argument(
        '--output_folder',
        required=True,
        help='Path to folder to write output to (this includes logs &  checkpoints).',
    )

    return parser.parse_args()


def train(output_folder, model, dataset, train):
    cpu_count = multiprocessing.cpu_count()
    train['num_workers'] = min(cpu_count, train['num_workers'])

    dataset_class = Datasets[dataset['dataset_type'].upper()].value

    train_data = [dataset_class(folder, dataset) for folder in train['training_folder']]
    train_data = train_data[0] if len(train_data) == 1 else ConcatDataset(train_data)
    validation_data = dataset_class(train['validation_folder'], dataset)
    
    trainer = Trainer(output_folder = output_folder,
                      train_data = train_data,
                      validation_data = validation_data,
                      model = model,
                      options = train)
    trainer.fit()

if __name__ == "__main__":
    parsed = vars(parse())
    jsons = {key: load_from_json(parsed[key]) for key in ['train', 'model', 'dataset']}
    jsons['output_folder'] = parsed['output_folder']
    train(**jsons)