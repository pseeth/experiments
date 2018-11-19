from torch.utils.data import Dataset
import pickle

class BaseDataset(Dataset):
    def __init__(self, folder, options=None):
        defaults = {
            'n_fft': 512,
            'hop_length': 128,
            'length': 400,
            'sample_rate': None,
            'output_type': 'psa',
            'cache': True,
            'fraction_of_dataset': 1.0,
            'weight_type': 'magnitude'
        }

        self.options = {**defaults, **(options if options else {})}
        self.files = self.get_files()

        if self.options['fraction_of_dataset'] < 1.0:
            num_files = int(len(self.files) * self.options['fraction_of_dataset'])
            shuffle(self.files)
            self.files = self.files[:num_files]

        self.files = self.files[:num_files]
        self.output_keys = ['log_spectrogram', 
                            'magnitude_spectrogram',
                            'source_spectrograms',
                            'assignments',
                            'weights',
                            'labels']
        self.init_specialized()

    def init_specialized(self):
        return

    def get_files(self):
        raise NotImplementedError()

    def load_audio_files(self, file_name):
        raise NotImplementedError()

    def __getitem__(self, i):
        _file = self.files[i]
        if self.create_cache:
            mix, sources, labels = self.load_audio_files(_file)
            output = self.construct_input_output(mix, sources)
            output['labels'] = labels
            self.write_to_cache(output, str(i))
        else:
            output = self.load_from_cache(str(i))
        return output

    def write_to_cache(self, data_dict, wav_file):
        with open(os.path.join(self.cache_location, wav_file), 'wb') as f:
            pickle.dump(data_dict, f)

    def load_from_cache(self, wav_file):
        with open(os.path.join(self.cache_location, wav_file), 'rb') as f:
            data = pickle.load(f)
        return data

    def whiten(self, data):
        data -= data.mean()
        data /= (data.std() + 1e-7)
        return data

    def construct_input_output(self, mix, sources, wav_file):
        log_spectrogram, mix_stft, _ = self.transform(mix, self.n_fft, self.hop_length)
        mix_magnitude, mix_phase = np.abs(mix_stft), np.angle(mix_stft)
        source_magnitudes = []
        source_log_magnitudes = []

        for source in sources:
            source_log_magnitude, source_stft, _ = self.transform(source, self.n_fft, self.hop_length)
            source_magnitude, source_phase = np.abs(source_stft), np.angle(source_stft)
            if self.output_type == 'msa':
                source_magnitude = np.minimum(mix_magnitude, source_magnitude)
            elif self.output_type == 'psa':
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

        output['weights'] = self.weight(output, self.options['weight_type'])

        return output


    def get_target_length_and_transpose(self, data_list, target_length):
        length = data_list[0].shape[1]
        if target_length == 'full':
            target_length = length
        if length > target_length:
            offset = np.random.randint(0, length - target_length)
        else:
            offset = 0

        for i, data in enumerate(data_list):
            pad_length = max(target_length - length, 0)
            pad_tuple = [(0, 0) for k in range(len(data.shape))]
            pad_tuple[1] = (0, pad_length)
            data_list[i] = np.pad(data, pad_tuple, mode='constant')

            if self.take_left_channel:
                data_list[i] = data_list[i][:, offset:offset + target_length, 0]
            else:
                data_list[i] = data_list[i][:, offset:offset + target_length, :self.num_channels]
            data_list[i] = np.swapaxes(data_list[i], 0, 1)

        return data_list

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

    def weight(data_dict, weight_type):
        weights = np.ones(data_dict['magnitude_spectrogram'].shape)
        if ('magnitude' in weight_type):
            weights = self.magnitude_weights(data['magnitude_spectrogram'])
        if ('threshold' in weight_type):
            weights = self.threshold_weights(data['log_spectrogram'])
        return weights

    @staticmethod
    def magnitude_weights(magnitude_spectrogram):
        weights = magnitude_spectrogram / (np.sum(magnitude_spectrogram))
        weights *= (magnitude_spectrogram.shape[0] * magnitude_spectrogram.shape[1])
        return weights

    @staticmethod
    def threshold_weights(log_spectrogram, threshold=-40):
        return ((log_spectrogram - np.max(log_spectrogram)) > threshold).astype(np.float32)