from torch.utils.data import Dataset
import pickle
import librosa
from scipy.io import wavfile
import numpy as np
import os
import shutil

class BaseDataset(Dataset):
    def __init__(self, folder, options):
        """This base class implements a variety of methods for loading source separation datasets such as WSJ0-[2,3]mix and datasets made with
        Scaper.
        
        Arguments:
            folder {str} -- Folder where dataset is contained.
        
        Keyword Arguments:
            options {dict} -- a dictionary containing the settings for the dataset loader. See config/defaults/metadata/dataset.json for full 
            description.
        """

        self.folder = folder
        self.files = self.get_files(self.folder)

        if self.options['cache']:
            self.cache_location = os.path.join(folder, 'cache', self.options['output_type'], self.options['weight_type'])
            shutil.rmtree(self.cache_location, ignore_errors=True)
            os.makedirs(self.cache_location, exist_ok=True)

        if self.options['fraction_of_dataset'] < 1.0:
            num_files = int(len(self.files) * self.options['fraction_of_dataset'])
            shuffle(self.files)
            self.files = self.files[:num_files]

    def get_files(self, folder):
        raise NotImplementedError()

    def load_audio_files(self, file_name):
        raise NotImplementedError()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        _file = self.files[i]
        if self.options['cache']:
            mix, sources, labels = self.load_audio_files(_file)
            output = self.construct_input_output(mix, sources)
            output['log_spectrogram'] = self.whiten(output['log_spectrogram'])
            output['labels'] = labels
            output = self.get_target_length_and_transpose(output, self.options['length'])
            self.write_to_cache(output, '%06d.pth' % i)
        else:
            output = self.load_from_cache('%06d.pth' % i)
        return output

    def write_to_cache(self, data_dict, file_name):
        with open(os.path.join(self.cache_location, file_name), 'wb') as f:
            pickle.dump(data_dict, f)

    def load_from_cache(self, wav_file):
        with open(os.path.join(self.cache_location, file_name), 'rb') as f:
            data = pickle.load(f)
        return data

    def whiten(self, data):
        data -= data.mean()
        data /= (data.std() + 1e-7)
        return data

    def construct_input_output(self, mix, sources):
        log_spectrogram, mix_stft, _ = self.transform(mix, self.options['n_fft'], self.options['hop_length'])
        mix_magnitude, mix_phase = np.abs(mix_stft), np.angle(mix_stft)
        source_magnitudes = []
        source_log_magnitudes = []

        for source in sources:
            source_log_magnitude, source_stft, _ = self.transform(source, self.options['n_fft'], self.options['hop_length'])
            source_magnitude, source_phase = np.abs(source_stft), np.angle(source_stft)
            if self.options['output_type'] == 'msa':
                source_magnitude = np.minimum(mix_magnitude, source_magnitude)
            elif self.options['output_type'] == 'psa':
                source_magnitude = np.maximum(0.0, np.minimum(mix_magnitude, source_magnitude * np.cos(source_phase - mix_phase)))
            source_magnitudes.append(source_magnitude)
            source_log_magnitudes.append(source_log_magnitude)

        source_magnitudes = np.stack(source_magnitudes, axis=-1)
        source_log_magnitudes = np.stack(source_log_magnitudes, axis=-1)

        shape = source_magnitudes.shape
        source_log_magnitudes = source_log_magnitudes.reshape(np.prod(shape[0:-1]), shape[-1])

        assignments = np.zeros(source_log_magnitudes.shape)
        source_argmax = np.argmax(source_log_magnitudes, axis=-1)
        assignments[np.arange(assignments.shape[0]), source_argmax] = 1.0
        assignments = assignments.reshape(shape)

        output = {
            'log_spectrogram': log_spectrogram,
            'magnitude_spectrogram': mix_magnitude,
            'assignments': assignments,
            'source_spectrograms': source_magnitudes,
        }
        output['weights'] = self.get_weights(output, self.options['weight_type'])
        return output


    def get_target_length_and_transpose(self, data_dict, target_length):
        length = data_dict['log_spectrogram'].shape[1]
        targets = ['log_spectrogram', 'magnitude_spectrogram', 'assignments', 'source_spectrograms', 'weights']
        if target_length == 'full':
            target_length = length
        if length > target_length:
            offset = np.random.randint(0, length - target_length)
        else:
            offset = 0

        for i, target in enumerate(targets):
            data = data_dict[target]
            pad_length = max(target_length - length, 0)
            pad_tuple = [(0, 0) for k in range(len(data.shape))]
            pad_tuple[1] = (0, pad_length)
            data_dict[target] = np.pad(data, pad_tuple, mode='constant')
            data_dict[target] = data_dict[target][:, offset:offset + target_length, :self.options['num_channels']]
            data_dict[target] = np.swapaxes(data_dict[target], 0, 1)

        return data_dict

    @staticmethod
    def transform(data, n_fft, hop_length):
        """Transforms multichannel audio signal into a multichannel spectrogram. 

        Arguments:
            data {[np.ndarray]} -- Audio signal of shape (n_channels, n_samples)
            n_fft {[int]} -- Number of FFT bins for each frame.
            hop_length {[int]} -- Hop between frames.
        
        Returns:
            [tuple] -- (log_spec, stft, n). log_spec contains the log_spectrogram,
            stft contains the complex spectrogram, and n is the original length of 
            the audio signal (used for reconstruction).
        """

        n = data.shape[-1]
        data = librosa.util.fix_length(data, n + n_fft // 2, axis=-1)
        stft = np.stack([librosa.stft(data[ch], n_fft=n_fft, hop_length=hop_length)
                         for ch in range(data.shape[0])], axis=-1)
        log_spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        return log_spectrogram, stft, n

    @staticmethod
    def invert_transform(data, n, hop_length):
        """Inverts a multichannel complex spectrogram back to the audio signal.
        
        Arguments:
            data {[np.ndarray]} -- Multichannel complex spectrogram
            n {int} -- Original length of audio signal
            hop_length {int} -- Hop length of STFT
        
        Returns:
            Multichannel audio signal
        """

        source = np.stack([librosa.istft(data[:, :, ch], hop_length=hop_length, length=n)
                           for ch in range(data.shape[-1])], axis=0)
        return source
    
    def get_weights(self, data_dict, weight_type):
        weights = np.ones(data_dict['magnitude_spectrogram'].shape)
        if ('magnitude' in weight_type):
            weights *= self.magnitude_weights(data_dict['magnitude_spectrogram'])
        if ('threshold' in weight_type):
            weights *= self.threshold_weights(data_dict['log_spectrogram'], self.options['weight_threshold'])
        return weights

    @staticmethod
    def magnitude_weights(magnitude_spectrogram):
        weights = magnitude_spectrogram / (np.sum(magnitude_spectrogram))
        weights *= (magnitude_spectrogram.shape[0] * magnitude_spectrogram.shape[1])
        return weights

    @staticmethod
    def threshold_weights(log_spectrogram, threshold=-40):
        return ((log_spectrogram - np.max(log_spectrogram)) > threshold).astype(np.float32)

    @staticmethod
    def load_audio(file_path):
        rate, audio = wavfile.read(file_path)
        if len(audio.shape) == 1:
            audio = np.expand_dims(audio, axis=-1)
        audio = audio.astype(np.float32, order='C') / 32768.0
        return audio.T, rate

    def inspect(self, i):
        import matplotlib.pyplot as plt
        from audio_embed import utilities

        input_data, mix_magnitude, output_data, _, _ = self[i]

        plt.style.use('dark_background')
        plt.figure(figsize=(20, 5))
        plt.imshow(input_data.T, aspect='auto', origin='lower')
        plt.show()

        mix, sources, one_hots = self.load_audio_files(self.files[i])
        print('Mixture')
        utilities.audio(mix, self.sr, ext='.wav')
        for j, source in enumerate(sources):
            print('Source %d' % j)
            utilities.audio(source, self.sr, ext='.wav')

        def mask_mixture(mask, mix):
            n = len(mix)
            mix = librosa.util.fix_length(mix, n + self.n_fft // 2)
            mix_stft = librosa.stft(mix, n_fft=self.n_fft, hop_length=self.hop_length)
            masked_mix = mix_stft * mask
            source = librosa.istft(masked_mix, hop_length=self.hop_length, length=n)
            return source

        for j in range(len(sources)):
            mask = (output_data[:, :, j] / (mix_magnitude + 1e-7)).T
            plt.figure(figsize=(20, 5))
            plt.subplot(121)
            plt.imshow(mask, aspect='auto', origin='lower')
            plt.subplot(122)
            log_output = librosa.amplitude_to_db(np.abs(output_data[:, :, j].T), ref=np.max)
            plt.imshow(log_output, aspect='auto', origin='lower')
            plt.show()
            isolated = mask_mixture(mask, mix)
            utilities.audio(isolated, self.sr, ext='.wav')
