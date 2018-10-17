import scaper
import os
import logging
import warnings
from utils import parallel_process, check_sources_not_equal_to_mix
import traceback

warnings.filterwarnings("ignore", category=scaper.scaper_warnings.ScaperWarning)
logging.getLogger('scaper').setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger('sox').setLevel(logging.ERROR)

def create_scaper_generator(foreground_directory, scene_duration, max_sources, background_directory=None):
    if background_directory is None:
        background_directory = foreground_directory
    sc = scaper.Scaper(scene_duration,
                       foreground_directory,
                       background_directory)

    sc.protected_labels = []
    sc.ref_db = -50
    sc.n_channels = 1
    sc.sr = 44100
    sc.min_silence_duration = None

    n_events = max_sources

    for n in range(n_events):
        sc.add_event(label=('choose', ['vocals', 'drums', 'bass', 'other']),
                     source_file=('choose', []),
                     source_time=('uniform', 0, 300),
                     event_time=('const', 0),
                     event_duration=('const', float(scene_duration)),
                     snr=('uniform', 25, 25),
                     pitch_shift=None,
                     time_stretch=None)
    return sc

def create_mixture(i, scene_duration, max_sources, foreground_directory, background_directory, target_directory):
    sc = create_scaper_generator(
        foreground_directory, scene_duration, max_sources, background_directory)
    audio_path = os.path.join(target_directory, '%06d.wav' % i)
    jams_path = os.path.join(target_directory, '%06d.json' % i)
    error = True
    while error:
        try:
            sc.generate(audio_path, jams_path, save_sources=True,
                        allow_repeated_label=False, reverb=0.0)
            error = check_sources_not_equal_to_mix(jams_path)
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            pass


base_directory = 'data/generated/musdb'
os.makedirs(base_directory, exist_ok=True)

num_training = 20000
num_validation = 2000
num_testing = 0
n_jobs = 20
splits = [('train', num_training),
          ('validation', num_validation),
          ('test', num_testing)]

directories = {
    'train': os.path.join('data/musdb', 'train'),
    'validation': os.path.join('data/musdb', 'validation'),
    'test': os.path.join('data/musdb', 'test'),
    'background': None
}

for split, num_split in splits:
    print("Generating %s" % split)
    target_directory = os.path.join(base_directory, split)
    os.makedirs(target_directory, exist_ok=True)
    mixes = [{"i": i,
              'scene_duration': 3.2,
              'max_sources': 4,
              'foreground_directory': directories[split],
              'background_directory': directories['background'],
              "target_directory": target_directory} for i in range(num_split)]

    parallel_process(mixes, create_mixture, n_jobs=n_jobs, use_kwargs=True)