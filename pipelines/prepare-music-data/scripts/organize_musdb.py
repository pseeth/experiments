import os
from random import shuffle
import random
import sox

random.seed(0)

splits = ['train', 'test']
sources = ['vocals', 'drums', 'bass', 'other']
num_validation = 14

def prep_directories():
    for split in ['train', 'validation', 'test']:
        for source in sources:
            os.makedirs(os.path.join('data', 'musdb', split, source), exist_ok=True)

prep_directories()

def copy_sources(song_paths, split):
    tfm = sox.Transformer()
    tfm.convert(n_channels=1)
    for song_path in song_paths:
        song_name = song_path.split('/')[-1]
        for source in sources:
            source_file = os.path.join(song_path, source + '.wav')
            destination_file = os.path.join('data', 'musdb', split, source, song_name + '.wav')
            print('Copying %s to %s' % (source_file, destination_file))
            tfm.build(source_file, destination_file)


for split in splits:
    source_path = os.path.join('data', 'raw', 'musdb', 'train' if split == 'validation' else split)
    song_names = [os.path.join(source_path, x) for x in os.listdir(source_path) if '.mp4' not in x]
    if split == 'train':
        shuffle(song_names)
        train_song_names = song_names[:-num_validation]
        copy_sources(train_song_names, split)

        validation_song_names = song_names[-num_validation:]
        copy_sources(validation_song_names, 'validation')
    else:
        test_song_name = song_names
        copy_sources(song_names, split)