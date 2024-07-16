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
    check_images = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dirs = 'block_output/88'
    save_dirs = 'simularity/88'
    stepdirs = []
    for index in range(steps, -1, -10):
        if index == 250:
            i = index - 1
        else:
            i = index 
        # images
        if check_images == True:

            output_dirs_img = output_dirs + '/imgs'
            output_dir = f'{output_dirs_img}/{i}'
            saved_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
            saved_files.sort()

            outputs_img = []
            for file_name in saved_files:
                file_path = os.path.join(output_dir, file_name)
                output = torch.load(file_path)
                outputs_img.append(output)

            final_output = outputs_img[-1]

            # concatenated_image = torch.cat(outputs_img, dim=2) 
            processed_tensors = []
            for tensor in outputs_img:
                first_batch_element = tensor[0]  # [8, 32, 32]
                split_tensor = torch.split(first_batch_element, 4, dim=0)[0]  # [4, 32, 32]
                processed_tensors.append(split_tensor)
            image = torch.stack(processed_tensors)

            from diffusers.models import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(f"pretrain/sd-vae-ft-ema").to(device)
            samples = vae.decode(image.to(device) / 0.18215).sample

            plot_path = './plot/417_baloon/imgs'
            os.makedirs(plot_path, exist_ok=True)
            save_image(samples, f"{plot_path}/concatenated_image_{i}.png", nrow=7, padding=2, normalize=True)
            print(f"Concatenated image saved to {plot_path}")

            # simularities
            mean_cosine_similarities = []
            std_cosine_similarities = []
            flag = 0
            for output in outputs_img:
                flag += 1
                
                output_flat = output.reshape(output.size(0), -1)
                final_output_flat = final_output.reshape(final_output.size(0), -1)
                
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
            plot_path = './plot/417_baloon/imgs/line'
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{i}.png")
            print(f"Plot saved to {plot_path}")
            plt.close()

        # block output
        if check_simularities == True:

            output_dir = f'{output_dirs}/res_1/{i}'
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
        
            class_mean_layers = []
            class_similarity = []
            flag = 0
            for output in outputs:
                flag += 1
                
                # output, _ = output.chunk(2, dim = 0) # Remove null class samples
                num_tensors = output.shape[0]
                output_flat = output.reshape(output.size(0), -1)
                mean_layer = output_flat.mean(dim=0, keepdim=True)
                norm_layer = torch.norm(mean_layer, p=2)
                diff_layer = output_flat - mean_layer
                class_mean_diff = torch.sum(torch.sum(diff_layer ** 2, dim = 1))
                class_mean_layers.append(class_mean_diff.item())

                if flag == 19 or flag == 1 or flag == 12 or flag == 28:
                    print(f'{flag} mean layer: {mean_layer[0]}')
                    print(f'{flag} norm layer: {norm_layer}')
                    print(f'{flag} diff layer: {diff_layer[0]}')
                    print(f'{flag} mean distance: {class_mean_diff}')
                    if flag == 28:
                        import sys
                        sys.exit()

                similarity_matrix = torch.zeros((num_tensors, num_tensors))

                for m in range(num_tensors):
                    for n in range(num_tensors):
                        similarity_matrix[m, n] = cosine_similarity(output_flat[m].unsqueeze(0), output_flat[n].unsqueeze(0), dim = 1)
                class_sim = (similarity_matrix.sum() - num_tensors) / (num_tensors ** 2 - num_tensors)
                class_similarity.append(class_sim.item())

                similarity_matrix_np = similarity_matrix.to('cpu').numpy()
                plt.figure(figsize=(20, 18))
                sns.heatmap(similarity_matrix_np, annot=True, cmap='coolwarm', vmin=0, vmax=1)
                plt.title('Image Similarity Matrix')
                plt.xlabel('Image Index')
                plt.ylabel('Image Index')
                plot_path = f'./plot/88/classsim/{i}'
                os.makedirs(plot_path, exist_ok=True)
                plt.savefig(f"{plot_path}/{flag-1}.png")
                # print(f"Plot saved to {plot_path}")
                plt.close()
                
            layers = np.arange(len(class_mean_layers))
            plt.figure(figsize=(10, 6))
            plt.plot(layers, class_mean_layers, marker='o', label='L2 Distance') # Distance between Feature and Class mean
            plt.xlabel('Layers')
            plt.ylabel('L2 Distance')
            plt.title('Distance between Feature and Class mean')
            plt.legend()
            plt.grid(True)
            plot_path = './plot/88/classmeandis'
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{i}.png")
            print(f"Plot saved to {plot_path}")
            plt.close()

            layers = np.arange(len(class_similarity))
            plt.figure(figsize=(10, 6))
            plt.plot(layers, class_similarity, marker='o', label='Mean Cos Similarity') 
            plt.xlabel('Layers')
            plt.ylabel('Cos Similarity')
            plt.title('Class Mean Similarity')
            plt.legend()
            plt.grid(True)
            plot_path = './plot/88/classmeancossim'
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{i}.png")
            print(f"Plot saved to {plot_path}")
            plt.close()


    # step_outputs = torch.cat(stepdirs)


if __name__ == "__main__":
    main()