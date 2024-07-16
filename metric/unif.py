import torch
import numpy as np

@torch.no_grad()
def lunif(features, t=2):
    """
    Compute the L_uniform loss for each layer.
    
    Args:
    features: torch tensor of shape (num_classes, num_layers, num_samples, feature_dim), features of each sample
    t: float, the scaling factor (default is 2)
    
    Returns:
    losses: torch tensor of shape (num_layers,), L_uniform loss for each layer
    """
    num_classes, num_layers, num_samples, feature_dim = features.shape
    device = features.device
    losses = torch.zeros(num_layers, device=device)

    
    for l in range(num_layers):
        # Flatten the features across classes and samples for the current layer
        layer_features = features[:, l, :, :].reshape(-1, feature_dim)
        
        # layer_features /= torch.norm(layer_features, p=2)
        # Compute the pairwise squared distances
        sq_pdist = torch.pdist(layer_features, p=2).pow(2)
        
        # Compute the exponential of the negative scaled distances
        exp_term = sq_pdist.mul(-t).exp()
        
        # Compute the mean of the exponential terms
        mean_exp_term = exp_term.mean() + 1e-8
        
        # Compute the log of the mean
        losses[l] = mean_exp_term.log()
    
    return losses