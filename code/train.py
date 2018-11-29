import multiprocessing
import os
from config.utils.defaults import load_from_json
from trainer import Trainer
import argparse

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
    return parser.parse_args()
    

def train(model, dataset, train):
    cpu_count = multiprocessing.cpu_count()
    train['num_workers'] = min(cpu_count, train['num_workers'])
    os.makedirs(train['output_folder'], exist_ok=True)



    print(f'Model: {model}\n')
    print(f'Dataset: {dataset}\n')
    print(f'Train: {train}\n')

if __name__ == "__main__":
    parsed = vars(parse())
    jsons = {key: load_from_json(val) for key, val in parsed.items()}
    train(**jsons)