import torch
import torch.nn as nn
import numpy as np

def weighted_cross_entropy(loss_function, spectrogram, source_masks, source_ibms, threshold):
    weights = spectrogram.view(spectrogram.shape[0], -1)
    weights -= weights.min(keepdim=True, dim=-1)[0]
    weights /= weights.max(keepdim=True, dim=-1)[0]
    weights = torch.sqrt((weights))
    if threshold is not None:
        weights[weights < threshold] = 0
    weights = weights.view(-1)
    losses = loss_function(source_masks.view(-1, source_masks.shape[-1]), 
           torch.argmax(source_ibms.view(-1, source_masks.shape[-1]), dim=-1).long())
    loss = (losses * weights).sum() / weights.sum()
    return loss

def affinity_cost(embedding, assignments):
    batch_size, num_points, embedding_size = embedding.size()
    _, _, _, num_sources = assignments.size()
    embedding = embedding.view(-1, embedding_size)
    assignments = assignments.view(-1, num_sources)

    silence_mask = torch.sum(assignments.detach(), dim=-1, keepdim=True)

    embedding = silence_mask * embedding
    class_weights = nn.functional.normalize(torch.sum(assignments.detach(), dim=-2), p=1, dim=-1).unsqueeze(0)
    class_weights = 1.0 / (torch.sqrt(class_weights) + 1e-7)
    weights = torch.matmul(assignments.detach(), class_weights.transpose(1, 0))
    norm = torch.sum(weights**2)**2
    
    assignments = assignments * weights.repeat(1, assignments.size()[-1])
    embedding = embedding * weights.repeat(1, embedding.size()[-1])

    embedding = embedding.view(batch_size, num_points, embedding_size)
    assignments = assignments.view(batch_size, num_points, num_sources)

    embedding_transpose = embedding.permute(0, 2, 1)
    assignments_transpose = assignments.permute(0, 2, 1)

    loss_est = torch.sum(torch.matmul(embedding_transpose, embedding)**2)
    loss_est_true = torch.sum(torch.matmul(embedding_transpose, assignments)**2)
    loss_true = torch.sum(torch.matmul(assignments_transpose, assignments)**2)
    loss = loss_est - 2*loss_est_true + loss_true
    loss = loss / norm.detach()
    return loss

def sparse_orthogonal_loss(attractors, weights=(1., 1.)):
    #l1_norm = torch.mean(torch.norm(attractors, dim=-1, p=1))
    non_overlap = torch.mean(torch.bmm(torch.abs(attractors), torch.abs(attractors).permute(0, 2, 1)))
    orth_loss = torch.mean(torch.bmm(attractors, attractors.permute(0, 2, 1)))
    return weights[0]*non_overlap + weights[1]*orth_loss

class PermutationInvariantLoss(nn.Module):
    def __init__(self, loss_function):
        super(PermutationInvariantLoss, self).__init__()
        self.loss_function = loss_function
        self.loss_function.reduce = False
        
    def forward(self, estimates, targets):
        num_batch, sequence_length, num_frequencies, num_sources = estimates.shape
        estimates = estimates.view(num_batch, sequence_length*num_frequencies, num_sources)
        targets = targets.view(num_batch, sequence_length*num_frequencies, num_sources)
        
        losses = []
        for p in permutations(range(num_sources)):
            loss = self.loss_function(estimates[:, :, list(p)], targets)
            loss = loss.mean(dim=1).mean(dim=-1)
            losses.append(loss)
        
        losses = torch.stack(losses,dim=-1)
        losses = torch.min(losses, dim=-1)[0]
        loss = torch.mean(losses)
        return loss

class WeightedL1Loss(nn.Module):
    def __init__(self, loss_function):
        super(WeightedL1Loss, self).__init__()
        self.loss_function = loss_function
        self.loss_function.reduce = False
        
    def forward(self, estimates, targets):
        shape = targets.shape
        weights = shape[-1]*nn.functional.normalize(1./torch.sum(estimates.view(shape[0], -1, shape[-1]) > 0, dim=1).float(), dim=-1, p=1).unsqueeze(1)
        loss = torch.mean(self.loss_function(estimates, targets).view(shape[0], -1, shape[-1]) * weights)
        return loss

