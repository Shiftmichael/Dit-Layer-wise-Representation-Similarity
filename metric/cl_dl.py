import torch
import numpy as np

@torch.no_grad()
def compute_cl_dl(features, feature_mean):
    """
    Compute C_l and D_l metrics for each layer.
    
    Args:
    features: torch tensor of shape (num_classes, num_layers, num_samples, feature_dim), features of each sample
    feature_mean: torch tensor of shape (num_classes, num_layers, feature_dim), mean features for each class
    num_classes: number of classes
    
    Returns:
    Cl: torch tensor of shape (num_layers,), C_l values for each layer
    Dl: torch tensor of shape (num_layers,), D_l values for each layer
    """
    num_classes, num_layers, num_samples, feature_dim = features.shape
    device = features.device

    # Compute the global mean
    global_mean = feature_mean.mean(dim=0)
    
    # Compute the trace of the within-class covariance matrix
    trace_Sw = torch.zeros(num_layers, device=device)
    
    for l in range(num_layers):
        for k in range(num_classes):  
            class_features = features[k, l]
            diff = class_features - feature_mean[k, l]
            trace_Sw[l] += (diff ** 2).sum() / (num_samples * num_classes) 
    
    # Compute the trace of the between-class covariance matrix
    trace_Sb = torch.zeros(num_layers, device=device)
    # for k in range(num_classes):
        # nk = num_samples  # Since we have num_samples per class
    for l in range(num_layers):
        for k in range(num_classes):
            diff = feature_mean[k, l] - global_mean[l]
            trace_Sb[l] += (diff ** 2).sum() / (num_classes)
    
    # Compute the C_l metric
    Cl = trace_Sw / trace_Sb
    
    # Compute the D_l metric
    Dl = torch.ones(num_layers, device=device)
    for l in range(num_layers):
        max_cosine_similarity = float('-inf')
        for k in range(num_classes):
            for k_prime in range(num_classes):
                if k != k_prime:
                    cosine_similarity = torch.dot(feature_mean[k, l], feature_mean[k_prime, l]) / \
                                        (torch.norm(feature_mean[k, l]) * torch.norm(feature_mean[k_prime, l]))
                    if cosine_similarity > max_cosine_similarity:
                        max_cosine_similarity = cosine_similarity
        Dl[l] -= max_cosine_similarity
    
    return Cl, Dl

@torch.no_grad()
def compute_cl_dl_return_all(features, feature_mean):
    
    num_classes, num_layers, num_samples, feature_dim = features.shape
    device = features.device

    # Compute the global mean
    global_mean = feature_mean.mean(dim=0)
    
    # Compute the trace of the within-class covariance matrix
    trace_Sw = torch.zeros(num_layers, device=device)
    for k in range(num_classes):
        for l in range(num_layers):
            class_features = features[k, l]
            diff = class_features - feature_mean[k, l]
            trace_Sw[l] += (diff ** 2).sum() / num_samples
    
    # Compute the trace of the between-class covariance matrix
    trace_Sb = torch.zeros(num_layers, device=device)
    for k in range(num_classes):
        nk = num_samples  # Since we have num_samples per class
        for l in range(num_layers):
            diff = feature_mean[k, l] - global_mean[l]
            trace_Sb[l] += nk * (diff ** 2).sum() / (num_classes * num_samples)
    
    # Compute the C_l metric
    Cl = trace_Sw / trace_Sb
    
    # Compute the D_l metric
    Dl = torch.ones(num_layers, device=device)
    for l in range(num_layers):
        # max_cosine_similarity = float('-inf')
        sum_cosine_similarity = 0
        cnt = 0
        for k in range(num_classes):
            for k_prime in range(num_classes):
                if k != k_prime:
                    cosine_similarity = torch.dot(feature_mean[k, l], feature_mean[k_prime, l]) / \
                                        (torch.norm(feature_mean[k, l]) * torch.norm(feature_mean[k_prime, l]))
                    # if cosine_similarity > max_cosine_similarity:
                    #     max_cosine_similarity = cosine_similarity
                    sum_cosine_similarity += cosine_similarity
                    cnt += 1
        # Dl[l] -= max_cosine_similarity
        Dl[l] -= sum_cosine_similarity / cnt
                    
    
    return Cl, Dl, trace_Sw, trace_Sb, global_mean

def compute_Sigma_W(features, class_means):
    num_classes, num_layers, num_samples, feature_dim = features.shape
    Sigma_W_per_layer = torch.zeros(num_layers, feature_dim, feature_dim)

    for l in range(num_layers):
        num_data = 0
        Sigma_W = torch.zeros(feature_dim, feature_dim).to(features.device)
        for k in range(num_classes):
            class_features = features[k, l, :, :]  # Shape: (num_samples, feature_dim)
            class_mean = class_means[k, l, :]  # Shape: (feature_dim,)
            for feature in class_features:
                diff = feature - class_mean
                Sigma_W += torch.outer(diff, diff)
                num_data += 1
        Sigma_W /= num_data
        Sigma_W_per_layer[l] = Sigma_W
    
    return Sigma_W_per_layer

def compute_Sigma_B(class_means, global_means):
    num_classes, num_layers, feature_dim = class_means.shape
    Sigma_B_per_layer = torch.zeros(num_layers, feature_dim, feature_dim)

    for l in range(num_layers):
        global_mean = global_means[l, :]  # Shape: (feature_dim,)
        Sigma_B = torch.zeros(feature_dim, feature_dim).to(class_means.device)
        for k in range(num_classes):
            class_mean = class_means[k, l, :]  # Shape: (feature_dim,)
            diff = class_mean - global_mean
            Sigma_B += torch.outer(diff, diff)
        Sigma_B /= num_classes
        Sigma_B_per_layer[l] = Sigma_B
    
    return Sigma_B_per_layer

@torch.no_grad()
def compute_cl(features):
    num_classes, num_layers, num_samples, feature_dim = features.shape

    # Compute class means
    class_means = torch.zeros(num_classes, num_layers, feature_dim).to(features.device)
    for k in range(num_classes):
        for l in range(num_layers):
            class_means[k, l, :] = torch.mean(features[k, l, :, :], dim=0)

    # Compute global means
    global_means = torch.mean(class_means, dim=0)

    # Compute Sigma_W and Sigma_B for each layer
    Sigma_W_per_layer = compute_Sigma_W(features, class_means)
    Sigma_B_per_layer = compute_Sigma_B(class_means, global_means)

    # Compute Cl for each layer
    Cl = torch.zeros(num_layers)
    for l in range(num_layers):
        Sw_trace = torch.trace(Sigma_W_per_layer[l])
        Sb_trace = torch.trace(Sigma_B_per_layer[l])
        Cl[l] = Sw_trace / Sb_trace
    
    return Cl