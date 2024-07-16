import torch
import numpy as np

# @torch.no_grad()
# def compute_cdnv(features):
#     """
#     Compute the Class-Distance Normalized Variance (CDNV) for each layer.
    
#     Args:
#     features: torch tensor of shape (num_classes, num_layers, num_samples, feature_dim), features of each sample
    
#     Returns:
#     cdnv: torch tensor of shape (num_layers,), CDNV for each layer
#     """
#     num_classes, num_layers, num_samples, feature_dim = features.shape
#     device = features.device
    
#     # Initialize tensor to hold the CDNV values for each layer
#     cdnv = torch.zeros(num_layers, device=device)
    
#     for l in range(num_layers):
#         layer_features = features[:, l, :, :]  # Shape: (num_classes, num_samples, feature_dim)
        
#         # Compute the mean for each class
#         class_means = layer_features.mean(dim=1)  # Shape: (num_classes, feature_dim)
        
#         # Compute the variance for each class
#         class_vars = ((layer_features - class_means.unsqueeze(1)) ** 2).mean(dim=1)  # Shape: (num_classes, feature_dim)
        
#         # Initialize accumulators for numerator and denominator
#         numerator = 0.0
#         denominator = 0.0
        
#         for i in range(num_classes):
#             for j in range(i + 1, num_classes):
#                 # Compute mean difference and its norm squared
#                 mean_diff = class_means[i] - class_means[j]
#                 mean_diff_norm_sq = torch.norm(mean_diff) ** 2
                
#                 # Compute variance terms
#                 var_i = class_vars[i].sum()
#                 var_j = class_vars[j].sum()
                
#                 # Update numerator and denominator
#                 numerator += var_i + var_j
#                 denominator += mean_diff_norm_sq
        
#         # Compute CDNV for the current layer
#         cdnv[l] = numerator / (denominator + 1e-8)  # Add a small constant to avoid division by zero
    
#     return cdnv

@torch.no_grad()
def compute_cdnv(features):
    """
    Compute CDNV for each layer.
    
    Args:
    features: torch tensor of shape (num_classes, num_layers, num_samples, feature_dim)
    
    Returns:
    cdnv_per_layer: torch tensor of shape (num_layers,)
    """
    num_classes, num_layers, num_samples, feature_dim = features.shape
    device = features.device
    
    # Initialize tensor to hold CDNV values for each layer
    cdnv_per_layer = torch.zeros(num_layers, device=device)
    
    for l in range(num_layers):
        # Initialize lists to hold variances and means of each class for this layer
        var_list = []
        mean_list = []
        
        # Calculate mean and variance for each class in this layer
        for k in range(num_classes):
            this_class_feature = features[k, l, :, :]  # Shape: (num_samples, feature_dim)
            mu_Q = torch.mean(this_class_feature, dim=0, keepdim=True)  # Calculate mean of this class
            class_var_all = torch.norm(this_class_feature - mu_Q, dim=1)**2  # Compute squared Euclidean distance from mean
            class_var = torch.mean(class_var_all)  # Average variance for this class
            mean_list.append(mu_Q)  # Append mean to list
            var_list.append(class_var)  # Append variance to list
        
        # Initialize list to hold CDNV values for this layer
        all_cdnv = []
        
        # Compute CDNV between all pairs of classes in this layer
        for i in range(len(var_list)):
            mean_Q1 = mean_list[i]
            var_Q1 = var_list[i]
            for j in range(i + 1, len(var_list)):
                mean_Q2 = mean_list[j]
                var_Q2 = var_list[j]
                cdnv = (var_Q1 + var_Q2) / (2 * torch.norm(mean_Q1 - mean_Q2) ** 2)  # Calculate CDNV
                all_cdnv.append(cdnv)
        
        # Calculate the mean CDNV value for this layer
        cdnv_per_layer[l] = torch.mean(torch.tensor(all_cdnv))
    
    return cdnv_per_layer
