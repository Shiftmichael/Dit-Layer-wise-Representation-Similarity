import torch
import numpy as np

@torch.no_grad()
def compute_nc1_per_layer(features, feature_mean):
    """
    Compute NC1 for each layer.
    
    Args:
    features: torch tensor of shape (num_classes, num_layers, num_samples, feature_dim)
    feature_mean: torch tensor of shape (num_classes, num_layers, feature_dim)
    
    Returns:
    nc1_per_layer: torch tensor of shape (num_layers,)
    """
    num_classes, num_layers, num_samples, feature_dim = features.shape
    device = feature_mean.device

    # Compute global mean for each layer
    global_mean = torch.mean(feature_mean, dim=0)

    # Initialize tensors for within-class and between-class covariance matrices
    Sw = torch.zeros((num_layers, feature_dim, feature_dim), dtype=features.dtype, device=device)
    Sb = torch.zeros((num_layers, feature_dim, feature_dim), dtype=features.dtype, device=device)

    # Compute within-class covariance matrix Sw and between-class covariance matrix Sb
    for k in range(num_classes):
        for l in range(num_layers):
            class_features = features[k, l, :, :]
            class_mean = feature_mean[k, l, :]
            
            # Within-class covariance for layer l
            centered_features = class_features - class_mean
            Sw[l] += torch.matmul(centered_features.T, centered_features) / num_samples
            
            # Between-class covariance for layer l
            mean_diff = (feature_mean[k, l, :] - global_mean[l, :]).view(-1, 1)
            Sb[l] += torch.matmul(mean_diff, mean_diff.T) / num_classes

    # Compute NC1 for each layer
    nc1_per_layer = torch.zeros(num_layers, dtype=features.dtype, device=device)
    for l in range(num_layers):
        Sb_inv = torch.linalg.pinv(Sw[l])  # Pseudoinverse of Sw
        trace_term = torch.trace(torch.matmul(Sw[l], Sb_inv))
        nc1_per_layer[l] = trace_term / num_classes

    return nc1_per_layer

@torch.no_grad()
def compute_nc2_per_layer(feature_mean):
    """
    Compute NC2 for each layer.
    
    Args:
    feature_mean: torch tensor of shape (num_classes, num_layers, feature_dim)
    
    Returns:
    nc2_per_layer: torch tensor of shape (num_layers,)
    """
    num_classes, num_layers, feature_dim = feature_mean.shape
    device = feature_mean.device

    # Compute global mean for each layer
    global_mean = torch.mean(feature_mean, dim=0)
    
    # Initialize tensor for NC2 values
    nc2_per_layer = torch.zeros(num_layers, dtype=feature_mean.dtype, device=device)
    
    for l in range(num_layers):
        # Compute the matrix M for the current layer
        M = (feature_mean[:, l, :] - global_mean[l, :]) / torch.norm(feature_mean[:, l, :] - global_mean[l, :], dim=1, keepdim=True)
        
        # Compute MM^T
        MM_T = torch.matmul(M, M.T)
        
        # Compute the empirical metric NC2
        identity_matrix = torch.eye(num_classes, dtype=feature_mean.dtype, device=device)
        one_vector = torch.ones((num_classes, 1), dtype=feature_mean.dtype, device=device)
        
        # Normalize MM_T by its Frobenius norm
        MM_T_normalized = MM_T / torch.norm(MM_T, p='fro')
        
        # Compute the target matrix
        target_matrix = (num_classes / (num_classes - 1)) * identity_matrix - (1 / (num_classes - 1)) * torch.matmul(one_vector, one_vector.T)
        
        # Normalize target matrix by its Frobenius norm
        target_matrix_normalized = target_matrix / torch.norm(target_matrix, p='fro')
        
        # Compute the NC2 value for the current layer
        nc2_per_layer[l] = torch.norm(MM_T_normalized - target_matrix_normalized, p='fro')

    return nc2_per_layer

def compute_nc3_per_layer(A, feature_mean):
    """
    Compute NC3 for each layer.
    
    Args:
    A: torch tensor of shape (num_layers, feature_dim, num_classes) - last-layer classifier
    feature_mean: torch tensor of shape (num_classes, num_layers, feature_dim)
    
    Returns:
    nc3_per_layer: torch tensor of shape (num_layers,)
    """
    num_classes, num_layers, feature_dim = feature_mean.shape
    device = feature_mean.device
    
    # Compute global mean for each layer
    global_mean = torch.mean(feature_mean, dim=0)
    
    # Initialize tensor for NC3 values
    nc3_per_layer = torch.zeros(num_layers, dtype=feature_mean.dtype, device=device)
    
    for l in range(num_layers):
        # Compute the matrix M for the current layer
        M = (feature_mean[:, l, :] - global_mean[l, :]) / torch.norm(feature_mean[:, l, :] - global_mean[l, :], dim=1, keepdim=True)
        
        # Compute A M^T
        A_layer = A[l]
        AMT = torch.matmul(A_layer, M.T)
        
        # Normalize AMT by its Frobenius norm
        AMT_normalized = AMT / torch.norm(AMT, p='fro')
        
        # Compute the target matrix
        identity_matrix = torch.eye(num_classes, dtype=feature_mean.dtype, device=device)
        one_vector = torch.ones((num_classes, 1), dtype=feature_mean.dtype, device=device)
        
        target_matrix = (1 / torch.sqrt(torch.tensor(num_classes - 1, dtype=feature_mean.dtype, device=device))) * (identity_matrix - (1 / num_classes) * torch.matmul(one_vector, one_vector.T))
        
        # Compute the NC3 value for the current layer
        nc3_per_layer[l] = torch.norm(AMT_normalized - target_matrix, p='fro')
    
    return nc3_per_layer