import argparse

def folders(parser: argparse.ArgumentParser):
    folders = parser.add_argument_group(
        'folders',
        description='Folder paths for training, validation & output',
    )

    folders.add_argument(
        'training_folder',
        nargs='*',
        help='Path(s) to folder(s) containing training data',
    )

    folders.add_argument(
        '--validation_folder',
        type=str.lower,
        help='Path to folder containing validation data',
    )

    folders.add_argument(
        '--output_folder',
        type=str.lower,
        help=(
            'Path to folder to write output to (this includes logs &'
            ' checkpoints)'
        ),
    )

def audio_processing(parser: argparse.ArgumentParser):
    audio_processing = parser.add_argument_group(
        'audio processing',
        # TODO: clarify wording
        description='Parameters for audio pre-processing',
    )

    audio_processing.add_argument(
        '--n_fft',
        type=int,
        default=256,
        # TODO: clarify wording
        help='Number of samples per fft (Fast Fourier Transform)',
    )

    audio_processing.add_argument(
        '--hop_length',
        type=int,
        default=64,
        # TODO: clarify wording
        help='Number of samples to shift fft (Fast Fourier Transofrm) window',
    )

    # TODO: mel projection size? is this in the right arg group?
    audio_processing.add_argument(
        '--projection_size',
        type=int,
        default=0,
        help='?', # TODO: clarify wording
    )

    audio_processing.add_argument(
        '--db_threshold',
        type=float,
        default=-80,
        help=(
            'Decibel (db) threshold to retain TF (time-frequency) bins. For'
            ' example, if given `10`, all bins with lower magnitudes will be'
            ' ignored.'
        ),
    )

def dataset(parser: argparse.ArgumentParser):
    dataset = parser.add_argument_group(
        'dataset',
        description='Parameters manipulating the given dataset',
    )

    dataset.add_argument(
        '--dataset_type',
        type=str.lower,
        default=['scaper'],
        choices=['scaper', 'wsj'],
        help=(
            'Labels identifying sources in your dataset. List only the labels'
            ' for sources you want to separate'
        ),
    )

    dataset.add_argument(
        '--source_labels',
        type=str.lower,
        nargs='+',
        help=(
            'Labels identifying sources in your dataset. List only the labels'
            ' for sources you want to separate'
        ),
    )

    dataset.add_argument(
        '--sample_strategy',
        default=['sequential'],
        type=str.lower,
        choices=['sequential', 'random'],
        help='Strategy for sampling training examples',
    )

    dataset.add_argument(
        '--group_sources',
        type=str.lower,
        nargs='+',
        action='append',
        help=(
            'Specify multiple source labels to treat them as one source. This'
            ' option requires specification of the `--source_labels` option'
            ' and any grouped sources *must* be listed in the values passed'
            ' to `--source_labels`. Multiple groupings are allowed, each with'
            ' separate usages of this flag. For example, if you have source '
            ' labels - bass, guitar, drums, violin, and vocals - and you would'
            ' like to treat vocals as one source, bass and guitar combined as'
            ' another, and drums and violin as a third:'
            ' `... --source_labels vocals bass guitar violin drums'
            ' --group_sources bass guitar --group_sources violin drums`.'
        ),
    )

def representation(parser: argparse.ArgumentParser):
    representation = parser.add_argument_group(
        'representation',
        description='Parameters specific to chosen representations',
    )

    # TODO: separate gaussian/covariance options to another argument group?

    representation.add_argument(
        '--num_gaussians_per_source',
        type=int,
        default=1,
        # TODO: clarify wording
        help='Number of gaussians to use to model each source',
    )

    representation.add_argument(
        '--covariance_type',
        default='diag',
        choices=['spherical', 'diag', 'tied_spherical', 'tied_diag'],
        help='Minimum covariance', # TODO: clarify wording
    )

    representation.add_argument(
        '--covariance_min',
        type=float,
        default=.5,
        help='Minimum covariance', # TODO: clarify wording
    )

    representation.add_argument(
        '--fix_covariance',
        action='store_true',
        help='Whether or not to fix covariance', # TODO: clarify wording
    )

def hyperparameters(parser: argparse.ArgumentParser):
    hyperparameters = parser.add_argument_group(
        'hyperparameters',
        description='Parameters to dictate training behavior',
    )

    hyperparameters.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help=(
            'Number of training epochs. One epoch means one run through all'
            ' given training data'
        ),
    )
    hyperparameters.add_argument(
        '--learning_rate', '-lr',
        type=float,
        default=1e-3,
        help='Weighting of backprop delta' # TODO: clarify wording
    )
    hyperparameters.add_argument(
        '--learning_rate_decay',
        type=float,
        default=.5,
        help=(
            # TODO: clarify wording
            'Rate at which to decay learning rate. A learning rate of .5'
            ' means the learning rate is halved every  <patience=5> epochs'
            ' that the change in the loss function is below some epsilon'
        )
    )
    hyperparameters.add_argument(
        '--patience',
        type=int,
        default=5,
        # TODO: clarify wording
        help=(
            'Number of epochs of minimal (within some epsilon) change in'
            ' loss function required before decaying learning rate'
        )
    )
    hyperparameters.add_argument(
        '--batch_size',
        type=int,
        default=5,
        # TODO: clarify wording
        help='Number of training samples per batch'
    )
    hyperparameters.add_argument(
        '--num_workers',
        type=int,
        default=5,
        help='?', # TODO: clarify wording
    )

    # TODO: make sure to lowercase given function and confirm it's valid,
    # validate that weight is a float, and validate target (limited set of
    # choices?)
    hyperparameters.add_argument(
        '--loss_function_classes_targets_weights', '-lfctw',
        type=str.lower, # lowercase choices (ignores numbers)
        nargs=3,
        action='append',
        metavar=('FUNCTION', 'TARGET', 'WEIGHT'),
        # TODO: clarify wording
        help=(
            'Triple of loss function, target model output on which to compute'
            " loss function and weight. `FUNCTION` may be any of the following:"
            " ['L1, 'DPCL', 'MSE', 'KL']. Multiple loss function triples may be"
            ' specified, each triple to its own `-lfctw` flag. E.g. to specify'
            ' two different loss functions:'
            '`... -lfctw L1 masks .4 -lfctw DPCL embeddings .6`'
        ),
    )

    hyperparameters.add_argument(
        '--optimizer',
        type=str.lower, # allow specification of choices with any casing
        default='adam',
        choices=['adam', 'rmsprop', 'sgd'],
        help='Optimizer for gradient descent', # TODO: clarify wording
    )

    hyperparameters.add_argument(
        '--clustering_type',
        default='kmeans',
        choices=['kmeans', 'gmm'], # TODO: what choices here?
        help='Type of clustering to perform on embeddings',
    )

    hyperparameters.add_argument(
        '--unfold_iterations',
        action='store_true',
        help='?', # TODO: clarify wording
    )

    hyperparameters.add_argument(
        '--activation_type',
        type=str.lower,
        default='sigmoid',
        choices=['sigmoid', 'relu'], # TODO: what choices here?
        help='?', # TODO: clarify wording
    )

    hyperparameters.add_argument(
        '--curriculum_learning',
        action='store_true',
        # TODO: clarify wording
        help='Whether or not to perform curriculum learning',
    )

    hyperparameters.add_argument(
        '--weight_method',
        type=str.lower,
        default='magnitude',
        choices=['magnitude'], # TODO: what choices here?
        help=(
            'Method by which to weight relative importance of accurately'
            ' predicting each TF (time-frequency) bin. Given `magnitude`,'
            ' training prioritizes accurately predicting louder bins.'
        ),
    )

    hyperparameters.add_argument(
        '--num_clustering_iterations',
        type=int,
        default=5,
        # TODO: clarify wording
        help='Number of iterations of clustering to perform',
    )

    hyperparameters.add_argument(
        '--initial_length',
        type=float,
        default=1.0,
        # TODO: clarify wording
        help='Fraction of initial length to use (for curriculum learning)',
    )

    # TODO: better location for this?
    hyperparameters.add_argument(
        '--target_type',
        type=str.lower,
        default='psa',
        choices=['psa', 'msa', 'ibm'],
        help='Mask approximation method', # TODO: clarify wording
    )

    hyperparameters.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='For L2 regularization', # TODO: clarify wording
    )

def miscellaneous(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--generate_plots',
        action='store_true',
        # TODO: clarify wording
        help='Whether or not to generate plots (of accuracy?)'
    )

    parser.add_argument(
        '--create_cache',
        action='store_true',
        help='Whether or not to cache ?' # TODO: clarify wording
    )

def toy_parser():
    parser = argparse.ArgumentParser(
        description=(
            'TUSSL: A framework for training deep net based audio'
            ' source separation'
        ),
        # TODO: also use `MetavarTypeHelpFormatter` somehow?
        formatter_class = argparse.ArgumentDefaultsHelpFormatter

    )

    subparsers = parser.add_subparsers(help='commands')
    subparsers_nested = subparsers.add_subparsers(help='nested')

    model = subparsers.add_parser('model')
    model.add_argument('--test')
    nested = subparsers_nested.add_parser('train')
    nested.add_argument('--nested-test')

    folders(parser)
    hyperparameters(parser)
    representation(parser)
    audio_processing(parser)
    dataset(parser)

    # TODO: post process parsed args to confirm valid loss_function triples
    return parser.parse_args()


if __name__ == '__main__':
    print(toy_parser())
