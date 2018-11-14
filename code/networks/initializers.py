import torch
import torch.nn as nn
import numpy as np

class AE(nn.Module):
    def __init__(self, num_attractors, embedding_size):
        super(AE, self).__init__()
        self.embedding_size = embedding_size
        self.num_attractors = num_attractors

        self.sequential = nn.Sequential(
            nn.Linear(self.num_attractors, self.embedding_size)
        )
        
    def initialize_parameters(self):
        for name, param in self.sequential.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
                param.requires_grad_(False)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
                param.data[int((self.embedding_size - 1)/2):-1, :] = np.sqrt(1.0)

    def forward(self, x):
        return self.sequential(x)
                
class CAE(nn.Module):
    def __init__(self, num_attractors, hidden_size, embedding_size):
        super(CAE, self).__init__()
        self.embedding_size = embedding_size
        self.num_attractors = num_attractors
        self.hidden_size = hidden_size

        self.sequential = nn.Sequential(
            nn.Linear(self.num_attractors + self.hidden_size*2, self.embedding_size)
        )

    def forward(self, inputs):
        one_hot = inputs[0]
        hidden = inputs[1]
        data = torch.cat([one_hot, hidden], dim=-1)
        return self.sequential(data)

class VAE(nn.Module):
    def __init__(self, num_attractors, embedding_size, latent_size=10):
        super(VAE, self).__init__()
        self.embedding_size = embedding_size
        self.num_attractors = num_attractors
        self.latent_size = latent_size

        self.fc1 = nn.Linear(self.num_attractors, self.latent_size)
        self.fc21 = nn.Linear(self.latent_size, self.embedding_size)
        self.fc22 = nn.Linear(self.latent_size, self.embedding_size)
        self.fc3 = nn.Linear(self.embedding_size, self.embedding_size)

        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_().requires_grad_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.fc3(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(x.shape[0], -1, self.num_attractors))
        z = self.reparameterize(mu, logvar)
        return self.decode(z)