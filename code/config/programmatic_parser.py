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
        if key == 'type':
            return eval(val)
        elif key == 'metavar':
            return tuple(val)

        return val

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
    no_positional_args = [
        key
        for key, val in all_metadata.items()
        if "is_positional" not in val or val["is_positional"] == False
    ]
    if set(all_defaults) != set(no_positional_args):
        print(f'In defaults, not no positional: {set(all_defaults) - set(no_positional_args)}')
        print(f'In no positional, not defaults: {set(no_positional_args) - set(all_defaults)}')
        raise Exception("Metadata keys do not match options keys")

    processed_metadata = {
        option_name: preprocess_metadata(
            option_name,
            metadata,
            all_defaults.get(option_name, None),
        )
        for option_name, metadata
        in all_metadata.items()
    }

    for option, metadata in processed_metadata.items():
        if option in all_defaults:
            metadata["default"] = all_defaults[option]

        subparser.add_argument(
            metadata.pop('flag'),
            **metadata # note that `flag` key has been popped by this point
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
            # TODO: handle option aliases
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
