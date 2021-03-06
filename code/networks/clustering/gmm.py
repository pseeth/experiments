import torch
import torch.nn as nn
import numpy as np
    
class GMM(nn.Module):
    def __init__(self, n_clusters, n_iterations=5, covariance_type='diag', covariance_min=0.0, fix_covariance=False):
        super(GMM, self).__init__()
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.covariance_min = covariance_min
        self.fix_covariance = fix_covariance
        
        allowed_covariance_types = ['diag', 'spherical', 'tied_diag', 'tied_spherical']
        if covariance_type not in allowed_covariance_types:
            raise ValueError('Covariance type must be one of [%s]' % (', '.join(allowed_covariance_types)))
        
        self.tied_covariance = 'tied' in covariance_type 
        
        if self.tied_covariance:
            self.covariance_type = covariance_type.split('_')[-1]
        else:
            self.covariance_type = covariance_type
            
            
    def initialize_parameters(self, data, var=1.0):
        sampled = data.new(data.shape[0], self.n_clusters).random_(0, data.shape[1])
        sampled = data.new(np.arange(0, data.shape[0])).unsqueeze(1).expand(-1, sampled.shape[1])*data.shape[1] + sampled
        sampled = sampled.long()
        means = torch.index_select(data.view(-1, data.shape[-1]), 0, sampled.view(-1)).view(data.shape[0], sampled.shape[-1], -1)
        
        var = data.new(data.shape[0], self.n_clusters, data.shape[-1]).fill_(var)
        pi = data.new(data.shape[0], self.n_clusters).fill_(1./self.n_clusters)
        
        return means, var, pi
        
    def update_parameters(self, posteriors, data, weights):
        if not isinstance(weights, float):
            weights = weights.unsqueeze(1).unsqueeze(-1)
            num_examples = weights.sum(dim=2)
        else:
            num_examples = data.shape[1]

        posteriors = posteriors.unsqueeze(-1).expand(-1, -1, -1, 1)
        posteriors = posteriors * weights #data weighting
        
        cluster_sizes = torch.sum(posteriors, dim=2)
        updated_pi = (cluster_sizes / num_examples).squeeze(-1)
        updated_pi = nn.functional.normalize(updated_pi, p=1, dim=-1)
        
        data = data.unsqueeze(1).expand(-1, 1, -1, -1)
        weighted_embeddings = posteriors * data
        updated_means = torch.sum(weighted_embeddings, dim=2) / (cluster_sizes + 1e-7)
        
        distance = data - updated_means.unsqueeze(2).expand(-1, -1, 1, -1)
        distance = posteriors * torch.pow(distance, 2)
        updated_var = torch.sum(distance, dim=2)

        if self.tied_covariance:
            updated_var = torch.sum(updated_var, dim=1, keepdim=True).expand(-1, updated_var.shape[1], -1)
            updated_var = (updated_var / (cluster_sizes.sum() + 1e-7))
        else:
            updated_var = updated_var / (cluster_sizes + 1e-7)

        if self.covariance_type == 'spherical':
            updated_var = torch.mean(updated_var, dim=-1, keepdim=True).expand(-1, -1, updated_var.shape[-1])
            
        return updated_means, updated_var, updated_pi
    
    def update_posteriors(self, likelihoods):
        #log-sum-exp trick https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        max_value = likelihoods.max(dim=1, keepdim=True)[0]
        likelihoods_sum = max_value + torch.log((likelihoods - max_value).exp().sum(dim=1, keepdim=True))
        posteriors = (likelihoods - likelihoods_sum).exp()
        return posteriors, likelihoods
    
    def update_likelihoods(self, data, means, var, pi):
        num_batch = data.shape[0]
        num_examples = data.shape[1]
        num_features = data.shape[-1]
        num_clusters = means.shape[1]
        inv_covariance =  1. / (var + 1e-6)
        data = data.unsqueeze(1).expand(-1, 1, -1, -1)
        means = means.unsqueeze(2).expand(-1, -1, 1, -1)
        distance = torch.pow(data - means, 2)
        distance = distance.view(num_batch*num_clusters, num_examples, num_features)
        inv_covariance = inv_covariance.view(num_batch*num_clusters, num_features, 1)
        
        distance = -.5 * torch.bmm(distance, inv_covariance)
        distance = distance.view(num_batch, num_clusters, num_examples)
        
        #log of determinant of 2*pi*variance, which is diagonal -> sum of log of 2 * pi * variance
        coeff = -.5 * torch.log(2 * np.pi * var).sum(dim=-1)
        prior = torch.log(pi)
        likelihoods = prior.unsqueeze(-1) + coeff.unsqueeze(-1) + distance
        return likelihoods
    
    def forward(self, data, parameters=None, weights=1.0):
        if parameters is None:
            parameters = self.initialize_parameters(data)
        means, var, pi = parameters

        for i in range(self.n_iterations):
            likelihoods = self.update_likelihoods(data, means, var, pi)
            posteriors, likelihoods = self.update_posteriors(likelihoods)
            means, var, pi = self.update_parameters(posteriors, data, weights)
            var = var + 1e-6 + self.covariance_min
            if self.fix_covariance:
                var[:, :, :] = self.covariance_min

        likelihoods = self.update_likelihoods(data, means, var, pi)
        posteriors, likelihoods = self.update_posteriors(likelihoods)
        return likelihoods.permute(0, 2, 1), posteriors.permute(0, 2, 1), (means, var, pi)