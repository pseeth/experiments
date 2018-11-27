# TODO: remove hack
import sys
sys.path.insert(0, "../utils")
from load import load_json
# TODO: remove hack

from builders import build_chimera_config, build_dpcl_config, build_mi_config

builders = {
    'chimera_recurrent': build_chimera_config,
    'dpcl_recurrent': build_dpcl_config,
    'mask_inference_recurrent': build_mi_config
}

def build(key, options=None, defaults=None): 
    config = builders[key](options)
    return config