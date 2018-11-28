from parser import parse

# TODO: remove hack
import sys
sys.path.insert(0, "./utils")
from defaults import load_from_json
# TODO: remove hack

def train():
    parsed = vars(parse())
    jsons = {key: load_from_json(val) for key, val in parsed.items()}
    train_model(**jsons)

def train_model(model, dataset, train):
    print(f'Model: {model}\n')
    print(f'Dataset: {dataset}\n')
    print(f'Train: {train}\n')

if __name__ == "__main__":
    train()