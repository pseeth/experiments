def build_dpcl_config(options=None):
    defaults = {
        'num_frequencies': 256,
        'num_mels': -1,
        'sample_rate': 44100,
        'hidden_size': 300,
        'bidirectional': True,
        'num_layers': 4,
        'embedding_size': 20,
        'dropout': .3,
        'embedding_activation': ['sigmoid', 'unitnorm'],
        'projection_trainable': False,
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
            'mel_projection': {
                'class': 'MelProjection',
                'args': {
                    'sample_rate': options['sample_rate'],
                    'num_frequencies': options['num_frequencies'],
                    'num_mels': options['num_mels'],
                    'direction': 'forward',
                    'trainable': options['projection_trainable'],
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
        },
        'connections': [
            ('mel_projection', ['log_spectrogram']),
            ('recurrent_stack', ['mel_projection']),
            ('embedding', ['recurrent_stack'])
        ],
        'output': ['embedding']
    }
    return config