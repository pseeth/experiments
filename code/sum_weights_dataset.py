import wsj_dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import trange

folder = '/pipeline/data/wsj0-mix/2speakers_anechoic/wav8k/min/tr/'


weight_sum = 0

dataset = wsj_dataset.WSJ0(folder=folder,
                           n_fft=256,
                           hop_length=64,
                           output_type='spatial_bootstrap',
                           weight_method='confidence',
                           length=400,
                           take_left_channel=False,
                           num_channels=1,
                           cache_location='test_conf')
dataset.create_cache = True

dataloader = DataLoader(dataset, batch_size=20, num_workers=10)
progress_bar = trange(len(dataset))

print('Bootstrap')

for (spectrogram, magnitude_spectrogram, source_spectrograms, source_ibms, weights, one_hots) in dataloader:
    progress_bar.set_description(str(weight_sum))
    progress_bar.update(20)
    _weight_sum = .5*(weights**2).sum().cpu().numpy()
    weight_sum += _weight_sum

bs_weight_sum = weight_sum
print(weight_sum)

print('Ground truth')

weight_sum = 0

dataset = wsj_dataset.WSJ0(folder=folder,
                           n_fft=256,
                           hop_length=64,
                           output_type='ibm',
                           weight_method='magnitude',
                           length=400,
                           take_left_channel=False,
                           num_channels=1,
                           cache_location='test_ibm')
dataset.create_cache = False

dataloader = DataLoader(dataset, batch_size=20, num_workers=10)
progress_bar = trange(len(dataset))


for (spectrogram, magnitude_spectrogram, source_spectrograms, source_ibms, weights, one_hots) in dataloader:
    progress_bar.set_description(str(weight_sum))
    progress_bar.update(20)
    _weight_sum = (weights**2).sum().cpu().numpy()
    weight_sum += _weight_sum

gt_weight_sum = weight_sum
print(bs_weight_sum, gt_weight_sum, bs_weight_sum / gt_weight_sum)