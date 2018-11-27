import argparse
import json

# TODO: add typing

def preprocess_metadata(option_name: str, metadata, default=None):
    """Massage data for splatting into `add_argument()`

    Currently performs two manipulations
        1. add `name` keyed to `(--)<option_name>`
        2. make type fields functions (unsafely using `eval()`, should fix).
            possible alternative - dictionary of functions keyed by 'type'

    Args:
        option_name - name of option
        metadata - dictionary of metadata related to option name

    Returns:
        massaged metadata
    """
    def process_single_key(key, val):
        return eval(key) if key == 'type' else val

    print(metadata)
    is_positional = "is_positional" in metadata and metadata["is_positional"]
    manual = {
        'flag': f"{'' if is_positional else '--'}{option_name}",
    }
    if default: manual['default'] = default

    return {
        **manual,
        **{
            key: process_single_key(key, val)
            for key, val in metadata.items()
            if key not in ['is_positional']
        }
    }

def add_arguments(subparser, defaults_path: str, metadata_path: str):
    with open(defaults_path) as defaults_f, open(metadata_path) as metadata_f:
        all_defaults = json.load(defaults_f)
        all_metadata = json.load(metadata_f)

    # could also just raise warning here
    # then iterate on intersection of keys later
    if set(all_defaults) != set(all_metadata):
        raise Exception("Metadata keys do not match options keys")

    processed_metadata = {
        option_name: preprocess_metadata(
            option_name,
            metadata,
            all_defaults[option_name],
        )
        for option_name, metadata
        in all_metadata.items()
    }

    for option, default in all_defaults.items():
        # guaranteed to succeed due to set comparison above
        metadata = processed_metadata[option]
        subparser.add_argument(
            metadata.pop('flag'),
            **metadata # note that `name` key has been popped by this point
        )

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'TUSSL: A framework for training deep net based audio'
            ' source separation'
        ),
        # TODO: also use `MetavarTypeHelpFormatter` somehow?
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers  = parser.add_subparsers()

    with open("./subparsers.json") as subparsers_file:
        for subparser_name, metadata in json.load(subparsers_file).items():
            subparser = subparsers.add_parser(
                subparser_name,
                formatter_class = argparse.ArgumentDefaultsHelpFormatter
            )
            add_arguments(
                subparser,
                metadata['defaults_path'],
                metadata['metadata_path'],
            )

    parser.parse_args()

if __name__ == "__main__":
    build_parser()
