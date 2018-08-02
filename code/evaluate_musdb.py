import musdb
import museval
import os
import torch
from utils import *
import librosa
from resampy import resample
import argparse
import json
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("--run_directory")
parser.add_argument("--track_id")
parser.add_argument("--estimate_dir", default='musdb_estimates')
parser.add_argument("--base_directory", default='/mm1/seetharaman')
args = parser.parse_args()
alpha = 1.0

mus = musdb.DB(root_dir='../data/raw/musdb/', is_wav=True)
run_directory = args.run_directory
estimate_directory = os.path.join(args.base_directory, args.estimate_dir, run_directory.split('/')[-1])

def evaluate(track, estimates_dir):
    print(track.name)
    museval._load_track_estimates(track, estimates_dir, estimates_dir)

tracknames = []
for _, folder, _ in os.walk(os.path.join(mus.root_dir, "test")):
    if len(folder) > 0:
        tracknames.append(folder)
tracknames = sorted(tracknames)[0]
track = mus.load_mus_tracks("test", tracknames=[tracknames[int(args.track_id)]])[0]
json_file = os.path.join(estimate_directory, "test", tracknames[int(args.track_id)] + '.json')
if not os.path.isfile(json_file):
    evaluate(track, estimate_directory)