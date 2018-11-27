from .base_dataset import BaseDataset
import librosa
import jams
import os
import scaper

class Scaper(BaseDataset):
    def __init__(self, folder, options=None):
        super(Scaper, self).__init__(folder, options)

        #initialization
        jam_file = self.files[0]
        jam = jams.load(jam_file)
        
        if len(self.options['source_labels']) == 0:
            all_classes = jam.annotations[0]['sandbox']['scaper']['fg_labels']
            classes = jam.annotations[0]['sandbox']['scaper']['fg_spec'][0][0][1]
            if len(classes) <= 1:
                classes = all_classes
            self.options['source_labels'] = classes
        
        if len(self.options['group_sources']) > 0 and 'group' not in self.options['source_labels']:
            self.options['source_labels'].append('group')
        self.options['source_indices'] = {source_name: i for i, source_name in enumerate(self.options['source_labels'])}

    def get_files(self, folder):
        files = [x for x in os.listdir(folder) if '.json' in x]
        return files        

    def load_audio_files(self, file_name):
        mix, sr = self.load_audio(file_name[:-4] + 'wav')
        jam = jams.load(file_name)
        data = jam.annotations[0]['data']['value']            
        classes = self.options['source_labels']

        sources = []
        one_hots = []
        group = []
        used_classes = []
        keep_columns = []

        for d in data:
            if d['role'] == 'foreground':
                source_path = d['saved_source_file']
                source_path = os.path.join(self.folder, source_path.split('/')[-1])
                sources.append(utils.load_audio(source_path)[0][0])
                one_hot = np.zeros(len(classes))
                one_hot[self.source_indices[d['label']]] = 1
                used_classes.append(d['label'])
                one_hots.append(one_hot)
                
                if d['label'] in self.options['group_sources'] or d['label'] in self.options['ignore_sources']:
                    group.append(sources[-1])
                    sources.pop()
                    one_hots.pop()
                    used_classes.pop()
                else:
                    keep_columns.append(self.source_indices[d['label']])

        if len(self.options['group_sources']) > 0:
            sources.append(sum(group))
            one_hot = np.zeros(len(classes))
            one_hot[self.source_indices['group']] = 1
            used_classes.append('group')
            one_hots.append(one_hot)
            keep_columns.append(self.source_indices['group'])

        source_order = [used_classes.index(c) for c in self.options['source_labels'] if c in used_classes]
        sources = [sources[i] for i in source_order]
        one_hots = [one_hots[i] for i in source_order]

        if self.group_sources:
            one_hots = np.stack(one_hots)[:, sorted(keep_columns)]
        else:
            one_hots = np.stack(one_hots)

        return mix, sources, one_hots