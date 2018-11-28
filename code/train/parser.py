import argparse

def parse():
    parser = argparse.ArgumentParser(
        description='Parse {model, dataset, train} JSONs',
    )

    parser.add_argument(
        'train',
        help='Path to JSON containing training configuration',
    )

    parser.add_argument(
        'model',
        help='Path to JSON containing model configuration',
    )

    parser.add_argument(
        'dataset',
        help='Path to JSON containing dataset configuration',
    )

    return parser.parse_args()
