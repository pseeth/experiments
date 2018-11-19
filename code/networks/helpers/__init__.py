from .build_chimera_config import build_chimera_config
from .build_dpcl_config import build_dpcl_config
from .build_mi_config import build_mi_config

builders = {
    'chimera_recurrent': build_chimera_config,
    'dpcl_recurrent': build_dpcl_config,
    'mask_inference_recurrent': build_mi_config
}

def build(key, options=None, defaults=None): 
    config = builders[key](options)
    return config