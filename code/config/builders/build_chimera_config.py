def build_chimera_config(options=None):
    defaults = {
        'num_frequencies': 256,
        'num_mels': -1,
        'sample_rate': 44100,
        'hidden_size': 300,
        'bidirectional': True,
        'num_sources': 2,
        'num_layers': 4,
        'embedding_size': 20,
        'dropout': .3,
        'embedding_activation': ['sigmoid', 'unitnorm'],
        'mask_activation': ['softmax'],
        'trainable': False,
        'rnn_type': 'lstm'
    }

    options = {**defaults, **(options if options else {})}
    options['num_features'] = (options['num_mels'] if options['num_mels'] > 0
                               else options['num_frequencies'])

    config = {
        'modules': {
            'log_spectrogram': {
                'input_shape': (-1, -1, options['num_frequencies'])
            },
            'magnitude_spectrogram': {
                'input_shape': (-1, -1, options['num_frequencies'])
            },
            'mel_projection': {
                'class': 'MelProjection',
                'args': {
                    'sample_rate': options['sample_rate'],
                    'num_frequencies': options['num_frequencies'],
                    'num_mels': options['num_mels'],
                    'direction': 'forward',
                    'trainable': options['trainable'],
                    'clamp': False
                }},
            'recurrent_stack': {
                'class': 'RecurrentStack',
                'args': {
                    'num_features': options['num_features'],
                    'hidden_size': options['hidden_size'],
                    'num_layers': options['num_layers'],
                    'bidirectional': options['bidirectional'],
                    'dropout': options['dropout'],
                    'rnn_type': options['rnn_type']
                }},
            'embedding': {
                'class': 'Embedding',
                'args': {
                    'num_features': options['num_features'],
                    'hidden_size': (2 * options['hidden_size'] if options['bidirectional']
                                    else options['hidden_size']),
                    'embedding_size': options['embedding_size'],
                    'activation': options['embedding_activation']
                }},
            'masks': {
                'class': 'Embedding',
                'args': {
                    'num_features': options['num_features'],
                    'hidden_size': (2 * options['hidden_size'] if options['bidirectional']
                                    else options['hidden_size']),
                    'embedding_size': options['num_sources'],
                    'activation': options['mask_activation']
                }},
            'inv_projection': {
                'class': 'MelProjection',
                'args': {
                    'sample_rate': options['sample_rate'],
                    'num_frequencies': options['num_frequencies'],
                    'num_mels': options['num_mels'],
                    'direction': 'backward',
                    'clamp': True
                }},
            'estimates': {
                'class': 'Mask',
                'args': {
                }}
        },
        'connections': [
            ('mel_projection', ['log_spectrogram']),
            ('recurrent_stack', ['mel_projection']),
            ('embedding', ['recurrent_stack']),
            ('masks', ['recurrent_stack']),
            ('inv_projection', ['masks']),
            ('estimates', ['inv_projection', 'magnitude_spectrogram'])
        ],
        'output': ['embedding', 'estimates']
    }
    return config