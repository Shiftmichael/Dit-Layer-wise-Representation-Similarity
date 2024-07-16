import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import cosine_similarity
import seaborn as sns
from torchvision.utils import save_image


@torch.no_grad()
def main():
    steps = 250
    check_simularities = True
    labels = [207, 360, 387, 974, 88, 979, 417]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dirs = 'block_output/classmean'
    save_dirs = 'simularity/417_baloon'
    for index in range(steps, -1, -10):
        if index == 250:
            i = index - 1
        else:
            i = index 

        # block output
        if check_simularities == True:
            classes_mean_layers = []

            for label in labels:    
                output_dir = f'{output_dirs}/{label}/{i}'
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
            
                class_mean_layers = []
                flag = 0
                for output in outputs:
                    flag += 1
                    
                    output, _ = output.chunk(2, dim = 0) # Remove null class samples
                    num_tensors = output.shape[0]
                    output_flat = output.reshape(output.size(0), -1)
                    mean_layer = output_flat.mean(dim=0, keepdim=True)
                    class_mean_layers.append(mean_layer)

                class_mean_layers = torch.cat(class_mean_layers)

                classes_mean_layers.append(class_mean_layers)
            
            classes_mean_layers = torch.stack(classes_mean_layers)

            similarity_matrices = []
            distance_sums = []
            for layer in range(28):
                features = classes_mean_layers[:, layer, :]  # Shape: [7, -1]

                # features = features / features.norm(dim=1, keepdim=True)  # Normalize the features
                # similarity_matrix = torch.matmul(features, features.T)  # Compute cosine similarity
                # similarity_matrices.append(similarity_matrix)

                # plt.figure(figsize=(10, 8))
                # sns.heatmap(similarity_matrix.to('cpu').numpy(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                # plt.title('Average Cosine Similarity Across 28 Layers')
                # plt.xlabel('Classes')
                # plt.ylabel('Classes')
                # plot_path = f'./plot/classmean/heatmaps/{layer}'
                # os.makedirs(plot_path, exist_ok=True)
                # plt.savefig(f"{plot_path}/{i}.png")
                # print(f"Plot saved to {plot_path}")
                # plt.close()

                distance_matrix = torch.cdist(features, features, p=2)  # Compute L2 distance
                # distance_sum = distance_matrix.sum(dim=0)  # Sum of distances for each class pair
                distance_sums.append(distance_matrix)
                plt.figure(figsize=(10, 8))
                sns.heatmap(distance_matrix.to('cpu').numpy(), annot=True, cmap='GnBu', vmax=7000)
                plt.title(f'L2 Distances {layer} Layer')
                plt.xlabel('Classes')
                plt.ylabel('Classes')
                plot_path = f'./plot/classmean/heatmaps/{layer}'
                os.makedirs(plot_path, exist_ok=True)
                plt.savefig(f"{plot_path}/{i}.png")
                print(f"Plot saved to {plot_path}")
                plt.close()

            average_distance_sum = torch.stack(distance_sums).mean(dim=0)
            average_distance_sum = average_distance_sum.to('cpu').numpy()
            plt.figure(figsize=(10, 8))
            sns.heatmap(average_distance_sum, annot=True, cmap='GnBu', vmax=7000)
            plt.title('Average L2 Distances Across 28 Layers')
            plt.xlabel('Classes')
            plt.ylabel('Classes')
            plot_path = './plot/classmean/heatmaps/mean'
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{i}.png")
            print(f"Plot saved to {plot_path}")
            plt.close()
            
            # average_similarity_matrix = torch.stack(similarity_matrices).mean(dim=0)
            # average_similarity_matrix = average_similarity_matrix.to('cpu').numpy()

            # plt.figure(figsize=(10, 8))
            # sns.heatmap(average_similarity_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            # plt.title('Average Cosine Similarity Across 28 Layers')
            # plt.xlabel('Classes')
            # plt.ylabel('Classes')
            # plot_path = './plot/classmean/heatmaps/mean'
            # os.makedirs(plot_path, exist_ok=True)
            # plt.savefig(f"{plot_path}/{i}.png")
            # print(f"Plot saved to {plot_path}")
            # plt.close()

    # step_outputs = torch.cat(stepdirs)


if __name__ == "__main__":
    main()