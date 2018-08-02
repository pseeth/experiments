import torch
import torch.nn as nn
import librosa
import numpy as np

class MaskEstimation(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers, num_sources, dropout, sample_rate=16000, projection_size=0, activation_type='sigmoid'):
        super(MaskEstimation, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_sources = num_sources
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation_type = activation_type
        self.sample_rate = sample_rate
        self.use_projection = projection_size > 0

        if self.use_projection:
            self.projection_size = projection_size
            self.add_module('projection', nn.Linear(input_size, self.projection_size))
            self.add_module('inverse_projection', nn.Linear(self.projection_size, input_size))
            self.original_input_size = self.input_size
            self.input_size = self.projection_size

        self.add_module('rnn', nn.LSTM(self.input_size, 
                          self.hidden_size, 
                          self.num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=self.dropout))

        self.add_module('linear', nn.Linear(self.hidden_size*2, self.input_size*self.num_sources))
        self.initialize_parameters()

    def initialize_parameters(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

        for name, param in self.linear.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
                
        if self.use_projection:
            mel_filters = librosa.filters.mel(self.sample_rate, 2*(self.original_input_size - 1), self.projection_size)
            mel_filters = (mel_filters.T / mel_filters.sum(axis=1)).T
            inverse_mel_filters = np.linalg.pinv(mel_filters)
            
            for name, param in self.projection.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                if 'weight' in name:
                    param.data = torch.from_numpy(mel_filters).float()
                param.requires_grad_(False)

            for name, param in self.inverse_projection.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                if 'weight' in name:
                    param.data = torch.from_numpy(inverse_mel_filters).float()
                param.requires_grad_(False)
                
    def learn_inverse_(self, learn=True):
        for name, param in self.inverse_projection.named_parameters():
            param.requires_grad_(learn) 

    def invert_projection(self, mask):
        mask = mask.permute(0, 1, 3, 2)
        mask = self.inverse_projection(mask)
        mask = mask.permute(0, 1, 3, 2)
        return mask
    
    def project(self, mask):
        mask = mask.permute(0, 1, 3, 2)
        mask = self.projection(mask)
        mask = mask.permute(0, 1, 3, 2)
        return mask

    def forward(self, input_data, one_hots=None):
        num_frequencies = input_data.shape[-1]
        if self.use_projection:
            input_data = self.projection(input_data)        
        num_batch, sequence_length, _ = input_data.size()
        
        output = self.rnn(input_data)[0]
        masks = self.linear(output)
        masks = masks.view(num_batch, sequence_length, -1, self.num_sources)
               
        if self.activation_type == 'sigmoid':
            masks = nn.functional.sigmoid(masks)
        elif self.activation_type == 'softmax':
            masks = nn.functional.softmax(masks, dim=-1)
        if self.use_projection:
            masks = self.invert_projection(masks)
            masks = masks.clamp(0.0, 1.0)
        return masks, None, None
