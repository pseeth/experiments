from builders import (
    build_chimera_config,
    build_dpcl_config,
    build_mask_inference_config
)
from parser import build_parser
import os

# TODO: remove hack
import sys
sys.path.insert(0, "./utils")
from defaults import save_to_json
# TODO: remove hack

def build_model(key, options=None):
    builders = {
        'chimera_recurrent': build_chimera_config,
        'dpcl_recurrent': build_dpcl_config,
        'mask_inference_recurrent': build_mask_inference_config,
    }
    return builders[key](options)

def config():
    _parser = build_parser()
    _parser.add_argument('--config_folder', default='.', type=str)
    args = vars(_parser.parse_args())
    if 'subparser' in args:
        subparser_name = args.pop('subparser')
        config_folder = args.pop('config_folder')
        os.makedirs(config_folder, exist_ok=True)
        save_to_json(
            f"{os.path.join(config_folder, subparser_name)}.json",
            (
                args
                if subparser_name in ['dataset', 'train']
                else build_model(subparser_name, args)
            )
        )

if __name__ == "__main__":
    config()
