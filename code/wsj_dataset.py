from torch.utils.data import Dataset
import librosa
import os
import numpy as np

class WSJ0(Dataset):
    def __init__(self, folder, length = 1.0, n_fft=512, hop_length=128, sr=None, num_channels=2, output_type='psa'):
        self.folder = folder
        self.wav_files = sorted([x for x in os.listdir(os.path.join(folder, 'mix')) if '.wav' in x])
        self.speaker_folders = sorted([x for x in os.listdir(folder) if 's' in x])

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.length = length
        self.sr = sr
        self.output_type = output_type
        self.stats = None
        self.whiten_data = False
        self.num_channels = num_channels

        wav_file = os.path.join(self.folder, 'mix', self.wav_files[0])
        mix, self.sr = librosa.load(wav_file, sr=self.sr, mono=False)
        self.channels_in_mix = mix.shape[0] if mix.shape[0] < 8 else int(mix.shape[0] / 2)


    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, i):
        wav_file = self.wav_files[i]
        mix, sources = self.load_wsj_mix(wav_file)
        input_data, mix_magnitude, source_magnitudes, source_ibm = self.construct_input_output(mix, sources)
        if self.whiten_data:
            input_data = self.whiten(input_data)
        return input_data, mix_magnitude, source_magnitudes, source_ibm, 0

    def whiten(self, data):
        if self.stats is None:
            data -= data.mean()
            data /= (data.std() + 1e-7)
        else:
            data -= self.stats['mean']
            data /= self.stats['std'] + 1e-7
        return data
    
    def construct_input_output(self, mix, sources):
        mix_log_magnitude, mix_stft, _ = self.transform(mix, self.n_fft, self.hop_length)
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

        source_ibm = np.zeros(source_log_magnitudes.shape)
        source_argmax = np.argmax(source_log_magnitudes, axis=-1)
        source_ibm[np.arange(source_ibm.shape[0]), source_argmax] = 1.0
        
        source_ibm = source_ibm.reshape(shape)
        silence_mask = np.expand_dims(mix_log_magnitude, axis=-1) > -50
        source_ibm = source_ibm * silence_mask

        if self.output_type == 'ibm':
            source_magnitudes = np.expand_dims(mix_magnitude, axis=-1) * source_ibm

        return mix_log_magnitude, mix_magnitude, source_magnitudes, source_ibm

    def load_wsj_mix(self, wav_file):
        sources = []
        channel_indices = np.arange(self.channels_in_mix)
        np.random.shuffle(channel_indices)
        channel_indices = channel_indices[:self.num_channels]

        for speaker in self.speaker_folders:
            speaker_path = os.path.join(self.folder, speaker, wav_file)
            mix_path = os.path.join(self.folder, 'mix', wav_file)

            mix, _ = librosa.load(mix_path, sr=self.sr, mono=False)
            source, _ = librosa.load(speaker_path, sr=self.sr, mono=False)

            mix = mix[channel_indices]
            source = source[channel_indices]
            sources.append(source)
        
        length_cutoff = int(mix.shape[0]*self.length)
        mix = mix[:length_cutoff]
        sources = [source[:length_cutoff] for source in sources]

        return mix, sources

    @staticmethod
    def transform(data, n_fft, hop_length):
        n = data.shape[-1]
        data = librosa.util.fix_length(data, n + n_fft // 2, axis=-1)
        stft = np.stack([librosa.stft(data[ch], n_fft=n_fft, hop_length=hop_length)
                         for ch in range(data.shape[0])], axis=-1)
        log_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        return log_spec, stft, n

    @staticmethod
    def invert_transform(data, n, hop_length):
        source = np.stack([librosa.istft(data[:, :, ch], hop_length=hop_length, length=n)
                           for ch in range(data.shape[-1])], axis=0)
        return source

    def inspect(self, i):
        import matplotlib.pyplot as plt
        from audio_embed import utilities

        input_data, mix_magnitude, output_data, _, _ = self[i]

        plt.style.use('dark_background')
        plt.figure(figsize=(20, 5))
        plt.imshow(input_data.mean(axis=-1), aspect='auto', origin='lower')
        plt.colorbar()
        plt.show()

        mix, sources = self.load_wsj_mix(self.wav_files[i])
        print('Mixture')
        utilities.audio(mix.T, self.sr, ext='.wav')
        for j, source in enumerate(sources):
            print('Source %d' % j)
            utilities.audio(source.T, self.sr, ext='.wav')

        def mask_mixture(mask, mix):
            _, mix_stft, n = self.transform(mix, n_fft=self.n_fft, hop_length=self.hop_length)
            masked_mix = mix_stft * mask
            source = self.invert_transform(masked_mix, hop_length=self.hop_length, n=n)
            return source

        for j in range(len(sources)):
            mask = (output_data[:, :, :, j] / (mix_magnitude + 1e-7))
            plt.figure(figsize=(20, 5))
            plt.subplot(121)
            plt.imshow(mask.mean(axis=-1), aspect='auto', origin='lower')
            plt.subplot(122)
            log_output = librosa.amplitude_to_db(np.abs(output_data[:, :, :, j]), ref=np.max)
            plt.imshow(log_output.mean(axis=-1), aspect='auto', origin='lower')
            plt.show()
            isolated = mask_mixture(mask, mix)
            utilities.audio(isolated.T, self.sr, ext='.wav')