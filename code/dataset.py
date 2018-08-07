from torch.utils.data import Dataset
import librosa
import jams
import os
import numpy as np
import scaper
import random
import sox

class ScaperLoader(Dataset):
    def __init__(self, folder, length = 1.0, n_fft=512, hop_length=128, sr=None, output_type='psa', group_sources=[], ignore_sources=[], source_labels=[]):
        self.folder = folder
        self.jam_files = sorted([os.path.join(folder, x) for x in os.listdir(folder) if '.json' in x])
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.length = length
        self.sr = sr
        self.output_type = output_type
        self.stats = None
        self.whiten_data = False
        self.group_sources = group_sources
        self.ignore_sources = ignore_sources
        self.source_labels = source_labels
        self.reorder_sources = False
        self.num_extra_sources = 1
        
        #initialization
        jam_file = self.jam_files[0]
        jam = jams.load(jam_file)
        
        if len(self.source_labels) == 0:
            all_classes = jam.annotations[0]['sandbox']['scaper']['fg_labels']
            classes = jam.annotations[0]['sandbox']['scaper']['fg_spec'][0][0][1]
            if len(classes) <= 1:
                classes = all_classes
            self.source_labels = classes
        
        if len(self.group_sources) > 0 and 'group' not in self.source_labels:
            self.source_labels.append('group')
        self.source_indices = {source_name: i for i, source_name in enumerate(self.source_labels)}
        _, self.sr = librosa.load(jam_file[:-4] + 'wav', sr=self.sr)     

    def __len__(self):
        return len(self.jam_files)

    def __getitem__(self, i):
        jam_file = self.jam_files[i]
        mix, sources, one_hots = self.load_jam_file(jam_file)
        input_data, mix_magnitude, source_magnitudes, source_ibm = self.construct_input_output(mix, sources)
        if self.whiten_data:
            input_data = self.whiten(input_data)
        return input_data, mix_magnitude, source_magnitudes, source_ibm, one_hots

    def whiten(self, data):
        if self.stats is None:
            data -= data.mean()
            data /= (data.std() + 1e-7)
        else:
            data -= self.stats['mean']
            data /= self.stats['std'] + 1e-7
        return data
    
    def construct_input_output(self, mix, sources):
        mix_log_magnitude, mix_stft = self.transform(mix, self.n_fft, self.hop_length)
        mix_magnitude, mix_phase = np.abs(mix_stft), np.angle(mix_stft)
        source_magnitudes = []
        source_log_magnitudes = []

        for source in sources:
            source_log_magnitude, source_stft = self.transform(source, self.n_fft, self.hop_length)
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
        source_log_magnitudes = source_log_magnitudes.reshape(shape)
        
        if self.output_type == 'ibm':
            source_magnitudes = np.expand_dims(mix_magnitude, axis=-1) * source_ibm
        
        #source_ratio = source_magnitudes / source_magnitudes.sum(axis=-1, keepdims=True)
        #source_magnitudes = np.expand_dims(mix_magnitude, axis=-1) * source_ratio
 
        return mix_log_magnitude, mix_magnitude, source_magnitudes, source_ibm

    def load_jam_file(self, jam_file):
        mix, sr = librosa.load(jam_file[:-4] + 'wav', sr=self.sr)
        
        jam = jams.load(jam_file)
        data = jam.annotations[0]['data']['value']            
        classes = self.source_labels

        sources = []
        one_hots = []
        group = []
        used_classes = []
        keep_columns = []

        for d in data:
            if d['role'] == 'foreground':
                source_path = d['saved_source_file']
                source_path = os.path.join(self.folder, source_path.split('/')[-1])
                sources.append(librosa.load(source_path, sr=self.sr)[0])
                one_hot = np.zeros(len(classes))
                one_hot[self.source_indices[d['label']]] = 1
                used_classes.append(d['label'])
                one_hots.append(one_hot)
                
                if d['label'] in self.group_sources or d['label'] in self.ignore_sources:
                    group.append(sources[-1])
                    sources.pop()
                    one_hots.pop()
                    used_classes.pop()
                else:
                    keep_columns.append(self.source_indices[d['label']])

        if len(self.group_sources) > 0:
            sources.append(sum(group))
            one_hot = np.zeros(len(classes))
            one_hot[self.source_indices['group']] = 1
            used_classes.append('group')
            one_hots.append(one_hot)
            keep_columns.append(self.source_indices['group'])
        
        if self.num_extra_sources > 0:
            num_sources = len(sources)
            shuffled = random.sample(classes, len(classes))
            for class_name in shuffled:
                if class_name not in used_classes:
                    if len(sources) >= num_sources + self.num_extra_sources:
                        break
                    one_hot = np.zeros(len(classes))
                    one_hot[classes.index(class_name)] = 1
                    one_hots.append(one_hot)
                    sources.append(np.zeros(sources[-1].shape))
                    used_classes.append(class_name)
        
        length_cutoff = int(mix.shape[0]*self.length)
        mix = mix[:length_cutoff]
        sources = [source[:length_cutoff] for source in sources]
        if self.reorder_sources:
            source_order = [used_classes.index(c) for c in self.source_labels if c in used_classes]
            sources = [sources[i] for i in source_order]
            one_hots = [one_hots[i] for i in source_order]
        if self.group_sources:
            one_hots = np.stack(one_hots)[:, sorted(keep_columns)]
        else:
            one_hots = np.stack(one_hots)
        return mix, sources, one_hots

    @staticmethod
    def transform(data, n_fft, hop_length):
        n = len(data)
        data = librosa.util.fix_length(data, n + n_fft // 2)
        stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length).T
        log_spec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        return log_spec, stft

    def inspect(self, i):
        import matplotlib.pyplot as plt
        from audio_embed import utilities

        input_data, mix_magnitude, output_data, _, _ = self[i]

        plt.style.use('dark_background')
        plt.figure(figsize=(20, 5))
        plt.imshow(input_data.T, aspect='auto', origin='lower')
        plt.show()

        mix, sources, one_hots = self.load_jam_file(self.jam_files[i])
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
