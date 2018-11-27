from builders import (
    build_chimera_config,
    build_dpcl_config,
    build_mask_inference_config
)
from parser import build_parser

# TODO: remove hack
import sys
sys.path.insert(0, "./utils")
from defaults import save_to_json
# TODO: remove hack

def build_model(key, options=None, defaults=None):
    builders = {
        'chimera_recurrent': build_chimera_config,
        'dpcl_recurrent': build_dpcl_config,
        'mask_inference_recurrent': build_mi_config
    }

    return builders[key](options)

def config():
    args = vars(build_parser())
    print(args)
    if args.get('subparser'):
        save_to_json(f'./{args.pop("subparser")}.json', args)

if __name__ == "__main__":
    config()
