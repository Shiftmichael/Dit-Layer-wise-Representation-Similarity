import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import cosine_similarity
import seaborn as sns

steps = 250

output_dirs = 'block_output/417_baloon'
stepdirs = []
for i in range(0, steps, 10):
    output_dir = f'{output_dirs}/{i}'
    saved_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    saved_files.sort()

    outputs = []
    for file_name in saved_files:
        file_path = os.path.join(output_dir, file_name)
        output = torch.load(file_path)
        outputs.append(output)
    
    for j, output in enumerate(outputs):
        print(f"Output {j} shape: {output.shape}")

    final_output = outputs[-1]

    stepdirs.append(final_output)
    
    mean_cosine_similarities = []
    std_cosine_similarities = []
    flag = 0
    for output in outputs:
        flag += 1
        
        output_flat = output.reshape(output.size(0), -1)
        final_output_flat = final_output.reshape(final_output.size(0), -1)

        if flag == 2: 
            print(torch.sum(torch.abs(output_flat - final_output_flat), dim = 1))
            check = torch.abs(output_flat - final_output_flat)
            print(f"check:{check[0]}, output:{output_flat[0]}")
            cos_sim = cosine_similarity(output_flat, final_output_flat, dim=1)
            print(cos_sim)
            mean_cos_sim = cos_sim.mean().item()
            print(mean_cos_sim)
            # sys.exit()
        
        cos_sim = cosine_similarity(output_flat, final_output_flat, dim=1)
        
        mean_cos_sim = cos_sim.mean().item()
        std_cos_sim = cos_sim.std().item()
        
        mean_cosine_similarities.append(mean_cos_sim)
        std_cosine_similarities.append(std_cos_sim) 
    
    layers = np.arange(len(mean_cosine_similarities))
    plt.figure(figsize=(10, 6))
    plt.plot(layers, mean_cosine_similarities, marker='o', label='Mean Cosine Similarity')
    plt.fill_between(layers, 
                    np.array(mean_cosine_similarities) - np.array(std_cosine_similarities), 
                    np.array(mean_cosine_similarities) + np.array(std_cosine_similarities), 
                    color='b', alpha=0.2, label='Std Dev')
    plt.xlabel('Layers')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity between Layers and Final Layer')
    plt.legend()
    plt.grid(True)
    plot_path = './plot/417_baloon/line'
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(f"{plot_path}/{i}.png")
    print(f"Plot saved to {plot_path}")
    plt.close()

    num_layers = len(outputs)
    cosine_sim_matrix = np.zeros((num_layers, num_layers))

    for k in range(num_layers):
        for j in range(num_layers):
            output_k_flat = outputs[k].reshape(outputs[k].size(0), -1)
            output_j_flat = outputs[j].reshape(outputs[j].size(0), -1)
            cos_sim = cosine_similarity(output_k_flat, output_j_flat, dim=1)
            cosine_sim_matrix[k, j] = cos_sim.mean().item()

    plt.figure(figsize=(20, 18))
    sns.heatmap(cosine_sim_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=[f'layer{i+1}' for i in range(num_layers)], yticklabels=[f'layer{i+1}' for i in range(num_layers)])
    plt.title('Cosine Similarity Heatmap between Layers')
    plt.xlabel('Layers')
    plt.ylabel('Layers')

    plot_path = 'plot/417_baloon/heatmap'
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(f"{plot_path}/{i}.png")
    print(f"Heatmap saved to {plot_path}")
    plt.close()

    
step_outputs = torch.cat(stepdirs)
