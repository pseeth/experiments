import torch
import torch.nn as nn
import librosa
import numpy as np
from .clustering import *
from .initializers import *

class DeepAttractor(nn.Module):
    def __init__(self, hidden_size=300, input_size=1025, num_layers=4, num_attractors=2, embedding_size=10, dropout=.3,
                 sample_rate=16000, projection_size=0, num_clustering_iterations=1, embedding_activation='tanh',
                 attractor_function_type='ae', normalize_embeddings=False, threshold=None, use_enhancement=False,
                 clustering_type='kmeans', covariance_type='diag', covariance_min=.5, fix_covariance=False,
                 use_likelihoods=False, num_gaussians_per_source=1):
        super(DeepAttractor, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.num_attractors = num_attractors
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_activation = embedding_activation
        self.use_projection = projection_size > 0
        self.sample_rate = sample_rate
        self.normalize_embeddings = normalize_embeddings
        self.threshold = threshold
        self.num_clustering_iterations = num_clustering_iterations
        self.attractor_function_type = attractor_function_type
        self.use_enhancement = use_enhancement
        self.clustering_type = clustering_type
        self.use_likelihoods = use_likelihoods
        self.num_gaussians_per_source = num_gaussians_per_source
        self.covariance_min =covariance_min
        self.fix_covariance = fix_covariance
        
        allowed_covariance_types = ['diag', 'spherical', 'tied_diag', 'tied_spherical']
        if covariance_type not in allowed_covariance_types:
            raise ValueError('Covariance type must be one of [%s]' % (', '.join(allowed_covariance_types)))
        
        self.covariance_type = covariance_type
        self.tied_covariance = 'tied' in covariance_type

        if self.use_projection:
            self.projection_size = projection_size
            self.add_module('projection', nn.Linear(input_size, self.projection_size))
            self.add_module('inverse_projection', nn.Linear(self.projection_size, input_size))
            self.original_input_size = self.input_size
            self.input_size = self.projection_size
        
        if self.clustering_type == 'kmeans':
            self.add_module('clusterer', KMeans(n_clusters=self.num_attractors, alpha=1.0, n_iterations=self.num_clustering_iterations))
            attractor_output_size = self.embedding_size
        elif self.clustering_type == 'gmm':
            self.add_module('clusterer', GMM(n_clusters=self.num_attractors, 
                                             n_iterations=self.num_clustering_iterations,
                                             covariance_type=self.covariance_type,
                                             covariance_min=self.covariance_min))
            #[means (embedding_size) variances (embedding_size), prior (1)]
            attractor_output_size = self.num_gaussians_per_source*(self.embedding_size*2 + 1)
        
        if self.attractor_function_type == 'ae':
            self.add_module('attractor_function', AE(self.num_attractors, attractor_output_size))
        elif self.attractor_function_type == 'vae':
            self.add_module('attractor_function', VAE(self.num_attractors, attractor_output_size))
        elif self.attractor_function_type == 'cae':
            self.add_module('attractor_function', CAE(self.num_attractors, self.hidden_size, attractor_output_size))
        
        self.add_module('rnn', nn.LSTM(self.input_size, 
                          self.hidden_size, 
                          self.num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=self.dropout))
        if self.use_enhancement:
            self.add_module('enhancement', nn.LSTM(2*self.input_size,
                                              self.input_size,
                                              int(self.num_layers / 2),
                                              batch_first=True,
                                              bidirectional=False,
                                              dropout=self.dropout))

        self.add_module('linear', nn.Linear(self.hidden_size*2, self.input_size*self.embedding_size))
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
                
        if self.clustering_type == 'gmm':
            self.attractor_function.initialize_parameters()
    
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

    def project_embedding_onto_attractors(self, embedding, attractors, weights):
        #Unfold clustering with class conditional parameters
        log_likelihoods, posteriors, attractors = self.clusterer(embedding, attractors, weights)
        if self.use_likelihoods:
            assignments = (log_likelihoods - log_likelihoods.max()).exp()
        else:
            assignments = posteriors
            
        assignments = assignments.view(assignments.shape[0], -1, self.input_size, assignments.shape[-1])
        assignments = assignments.view(assignments.shape[0], assignments.shape[1], self.input_size, -1, self.num_gaussians_per_source)
        assignments = assignments.sum(dim=-1)
        
        if self.use_enhancement:
            masks = self.enhance(weights, assignments)
        else:
            masks = assignments
                
        return masks, attractors, assignments, log_likelihoods
    
    def generate_attractors(self, inputs):
        if inputs is None:
            attractors = None
        else:
            attractors = self.attractor_function(inputs)
            
            if self.clustering_type == 'gmm':
                attractors = attractors.view(attractors.shape[0], attractors.shape[1]*self.num_gaussians_per_source, -1)
                means, var, pi = torch.split(attractors, [self.embedding_size, self.embedding_size, 1], dim=-1)
                means = self.activation(means)
                
                var = var ** 2
                if 'spherical' in self.covariance_type:
                    var = var.mean(dim=-1, keepdim=True).expand(-1, -1, self.embedding_size)
                if self.tied_covariance:
                    var = var.mean(dim=1, keepdim=True).expand(-1, attractors.shape[1], -1)
                if self.fix_covariance:
                    var[:, :, :] = self.covariance_min
                pi = nn.functional.softmax(pi.squeeze(-1), dim=-1)
                attractors = (means, var, pi)
            else:
                attractors = self.activation(attractors)

        return attractors
    
    def activation(self, data):
        if self.embedding_activation == 'sigmoid':
            data = nn.functional.sigmoid(data)
        elif self.embedding_activation == 'tanh':
            data= nn.functional.tanh(data)
        if self.normalize_embeddings:
            data = nn.functional.normalize(data, dim=-1, p=2)
        return data
            
    def enhance(self, mixture, assignments):
        mixture = mixture.view(assignments.shape[:-1])
        num_batch, sequence_length, num_features, num_sources = assignments.shape
        mixture = mixture.unsqueeze(-1).expand(-1, -1, -1, assignments.size(-1))
        
        enhancement_input = torch.cat([mixture, assignments], dim=-2)
        enhancement_input = enhancement_input.permute(0, 3, 2, 1).contiguous()
        enhancement_input = enhancement_input.view(-1, sequence_length, 2*num_features)
        
        enhanced = self.enhancement(enhancement_input)[0].contiguous()
        enhanced = enhanced.view(num_batch, num_sources, sequence_length, num_features)
        enhanced = enhanced.permute(0, 2, 3, 1)
        return enhanced

    def forward(self, input_data, one_hots=None):
        num_frequencies = input_data.shape[-1]
        if self.use_projection:
            input_data = self.projection(input_data)
            
        num_batch, sequence_length, _ = input_data.size()
                
        weights = input_data.view(num_batch, -1).clone()
        weights -= weights.min(keepdim=True, dim=-1)[0]
        weights /= (weights.max(keepdim=True, dim=-1)[0] + 1e-7)
        self.weights = weights
                
        if self.threshold is not None:
            weights[weights < self.threshold] = 0.0
            weights[weights > self.threshold] = 1.0
        
        output = self.rnn(input_data)[0]
        embedding = self.linear(output)
        embedding = embedding.view(num_batch, -1, self.embedding_size)
        embedding = self.activation(embedding)
        self.embedding = embedding
                
        if one_hots is None or self.attractor_function_type != 'cae':
            attractors = self.generate_attractors(one_hots)
        elif self.attractor_function_type == 'cae':
            attractors = self.generate_attractors([one_hots, output.mean(dim=1, keepdim=True).expand(-1, self.num_attractors, -1)])
        
        masks, attractors, assignments, log_likelihoods = self.project_embedding_onto_attractors(embedding, attractors, weights)
        
        self.attractors = attractors
        self.assignments = assignments
        masks = masks.view(num_batch, sequence_length, -1, masks.shape[-1])
        
        if self.use_projection:
            masks = self.invert_projection(masks)
            masks = masks.clamp(0.0, 1.0)
            
        return masks, attractors, embedding, log_likelihoods