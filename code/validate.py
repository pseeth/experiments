import torch
import utils
from torch.utils.data import DataLoader
from tqdm import trange
import random
import numpy as np

def validate(model, dset, writer, n_iter, params, device, loss_function, num_validation=5):
    model.eval()
    sample_indices = random.sample(range(len(dset)), num_validation)
    with torch.no_grad():
        for sample in sample_indices:
            n_iter += int(n_iter / num_validation + 1)
            mix, sources, one_hots = dset.load_jam_file(dset.jam_files[sample])
            spectrogram, magnitude_spectrogram, source_spectrograms, source_ibm = dset.construct_input_output(mix, sources)
            spectrogram = dset.whiten(spectrogram)
            num_frequencies = spectrogram.shape[-1]

            original_shape = spectrogram.shape
            writer.add_audio('audio/mix', torch.from_numpy(mix), n_iter, sample_rate = dset.sr)

            spectrogram = torch.from_numpy(spectrogram).unsqueeze(0).requires_grad_().to(device)
            one_hots = torch.from_numpy(one_hots).unsqueeze(0).float().requires_grad_().to(device)
            masks = model(spectrogram, one_hots)[0].cpu().squeeze(0).data.numpy()

            for j in range(0, masks.shape[-1]):
                mask = masks[:, :, j]
                isolated = utils.mask_mixture(mask.T, mix, dset.n_fft, dset.hop_length)
                writer.add_audio('audio/source_%02d' % j, torch.from_numpy(isolated), n_iter, sample_rate = dset.sr)

                mask = source_spectrograms[:, :, j] / (magnitude_spectrogram + 1e-7)
                isolated = utils.mask_mixture(mask.T, mix, dset.n_fft, dset.hop_length)
                writer.add_audio('audio/gt_source_%02d' % j, torch.from_numpy(isolated), n_iter, sample_rate = dset.sr)

                mask = source_ibm[:, :, j]
                isolated = utils.mask_mixture(mask.T, mix, dset.n_fft, dset.hop_length)
                writer.add_audio('audio/binary_source_%02d' % j, torch.from_numpy(isolated), n_iter, sample_rate = dset.sr)

            images = []

            def prepare_image(image):
                image = 20*np.log10(image + 1e-4)
                image -= -80
                image /= (image.max() + 1e-4)
                return image

            for k in range(source_spectrograms.shape[-1]):
                images.append(np.flipud(source_spectrograms[:, :, k].reshape(-1, num_frequencies).T).copy())
                images[-1] = prepare_image(images[-1])
                images.append(np.flipud(source_ibm[:, :, k].reshape(-1, num_frequencies).T).copy())
                images.append(np.flipud((masks[:, :, k] * magnitude_spectrogram).reshape(-1, num_frequencies).T).copy())
                images[-1] = prepare_image(images[-1])

            images.append(np.flipud(magnitude_spectrogram.reshape(-1, num_frequencies).T).copy())
            images[-1] = prepare_image(images[-1])
            summary_image = np.hstack(images)

            writer.add_image('spectrograms/summary', torch.from_numpy(summary_image), n_iter)

        dset_loader = DataLoader(dset, batch_size=params['batch_size'], num_workers=params['num_workers'])
        progress_bar = trange(len(dset_loader))
        val_loss = []

        for (spectrogram, magnitude_spectrogram, source_spectrograms, source_ibms, one_hots) in dset_loader:
            spectrogram = spectrogram.to(device).requires_grad_()
            magnitude_spectrogram = magnitude_spectrogram.to(device).unsqueeze(-1).requires_grad_()
            source_spectrograms = source_spectrograms.float().to(device)
            source_ibms = source_ibms.to(device)
            one_hots = one_hots.float().to(device).requires_grad_()

            source_masks = model(spectrogram, one_hots)[0]

            source_estimates = source_masks * magnitude_spectrogram
            loss = loss_function(source_estimates, source_spectrograms)

            progress_bar.set_description(str(loss.item()))
            progress_bar.update(1)
            val_loss.append(loss.item())

        val_loss = np.mean(val_loss)
    model.train()

    return val_loss