import argparse
import json
from pprint import pprint

# TODO: add typing

def preprocess_metadata(option_name: str, metadata):
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

    is_positional = option_name in metadata and metadata[option_name]
    return {
        'flag': f"{'' if is_positional else '--'}{option_name}",
        **{
            process_single_key(key, val)
            for key, val in metadata.items()
            if key not in ['positional']
        }
    }

def build_parser():
    with open('./defaults/models/dpcl_recurrent.json') as f:
        dpcl_options = json.load(f)

    with open('./defaults/models/metadata/dpcl_recurrent.json') as f:
        dpcl_metadata = json.load(f)

    # could also just raise warning here
    # then iterate on intersection of keys later
    if set(dpcl_options) != set(dpcl_metadata):
        raise Exception("Metadata does not match options")

    metadata_dict = {
        option_name: preprocess_metadata(option_name, metadata)
        for option_name, metadata
        in dpcl_metadata.items()
    }

    parser = argparse.ArgumentParser(
        description=(
            'TUSSL: A framework for training deep net based audio'
            ' source separation'
        ),
        # TODO: also use `MetavarTypeHelpFormatter` somehow?
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    for option_name, default in dpcl_options.items():
        # guaranteed to succeed due to set comparison above
        metadata = metadata_dict[option_name]
        parser.add_argument(
            metadata.pop('flag'),
            **metadata # note that `name` key has been popped by this point
        )

    parser.parse_args()

if __name__ == "__main__":
    build_parser()
