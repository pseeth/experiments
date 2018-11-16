from torch import nn
import json
from . import modules

class SeparationModel(nn.Module):
    def __init__(self, config):
        """
        SeparationModel takes a configuration file or dictionary that describes the model
        structure, which is some combination of MelProjection, Embedding, RecurrentStack,
        ConvolutionalStack, and other modules found in modules.py. The configuration file
        can be built using the helper functions in networks.helpers:
            - build_dpcl_config: Builds the original deep clustering network, mapping each
                time-frequency point to an embedding of some size. Takes as input a
                log_spectrogram.
            - build_mi_config: Builds a "traditional" mask inference network that maps the mixture
                spectrogram to source estimates.  Takes as input a log_spectrogram and a
                magnitude_spectrogram.
            - build_chimera_config: Builds a Chimera network with a mask inference head and a
                deep clustering head to map. A combination of MI and DPCL. Takes as input a
                log_spectrogram and a magnitude_spectrogram.

        References:
            Hershey, J. R., Chen, Z., Le Roux, J., & Watanabe, S. (2016, March).
            Deep clustering: Discriminative embeddings for segmentation and separation.
            In Acoustics, Speech and Signal Processing (ICASSP),
            2016 IEEE International Conference on (pp. 31-35). IEEE.

            Luo, Y., Chen, Z., Hershey, J. R., Le Roux, J., & Mesgarani, N. (2017, March).
            Deep clustering and conventional networks for music separation: Stronger together.
            In Acoustics, Speech and Signal Processing (ICASSP),
            2017 IEEE International Conference on (pp. 61-65). IEEE.

        Args:
            config: (str, dict) Either a config dictionary built from one of the helper functions,
                or the path to a json file containing a config built from the helper functions.

        Examples:
            >>> args = {
            >>>    'num_frequencies': 512,
            >>>    'num_mels': 128,
            >>>    'sample_rate': 44100,
            >>>    'hidden_size': 300,
            >>>    'bidirectional': True,
            >>>    'num_layers': 4,
            >>>    'embedding_size': 20,
            >>>    'num_sources': 4,
            >>>    'embedding_activation': ['sigmoid', 'unitnorm'],
            >>>    'mask_activation': ['softmax']
            >>> }
            >>> config = helpers.build_chimera_config(args)
            >>> with open('config.json', 'w') as f:
            >>>    json.dump(config, f)
            >>> model = SeparationModel('chimera_config.json')
            >>> test_data = np.random.random((1, 100, 512))
            >>> data = torch.from_numpy(test_data).float()
            >>> output = model({'log_spectrogram': data,
            >>>                'magnitude_spectrogram': data})

        """
        super(SeparationModel, self).__init__()
        if type(config) is str:
            with open(config, 'r') as f:
                config = json.load(f)
        module_dict = {}
        self.input = {}
        for module_key in config['modules']:
            module = config['modules'][module_key]
            if 'input_shape' not in module:
                class_func = getattr(modules, module['class'])
                module_dict[module_key] = class_func(**module['args'])
            else:
                self.input[module_key] = module['input_shape']

        self.layers = nn.ModuleDict(module_dict)
        self.connections = config['connections']
        self.output_keys = config['output']

    def forward(self, data):
        """
        Args:
            data: (dict) a dictionary containing the input data for the model. Should match the input_keys
                in self.input.

        Returns:

        """
        if not all(name in list(data) for name in list(self.input)):
            raise ValueError("Not all keys present in data! Needs {}".format(', '.join(self.input)))
        output = {}
        for connection in self.connections:
            layer = self.layers[connection[0]]
            input_data = []
            for c in connection[1]:
                if c in self.input:
                    input_data.append(data[c])
                else:
                    input_data.append(output[c])
            output[connection[0]] = layer(*input_data)
        output = {o: output[o] for o in self.output_keys}
        return output