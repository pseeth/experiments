from parser import parse
import multiprocessing
import os
from networks import SeparationModel

cpu_count = multiprocessing.cpu_count()
from config.utils.defaults import load_from_json

def parse_jsons():
    parsed = vars(parse())
    jsons = {key: load_from_json(val) for key, val in parsed.items()}
    return jsons

def train(model, dataset, train):
    train['num_workers'] = min(cpu_count, train['num_workers'])
    os.makedirs(train['output_folder'], exist_ok=True)


    print(f'Model: {model}\n')
    print(f'Dataset: {dataset}\n')
    print(f'Train: {train}\n')

if __name__ == "__main__":
    jsons = parse_jsons()
    train(**jsons)