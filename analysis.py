import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import cosine_similarity
import seaborn as sns
from torchvision.utils import save_image

def unpatchify(x, patch_size, out_channels):
    c = out_channels
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs

@torch.no_grad()
def main():
    steps = 250
    patch_size = 2
    out_channels = 8
    check_simularities = True
    check_images = True
    images_num = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dirs = 'block_output/282'
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
                first_batch_element = tensor[images_num]  # [8, 32, 32]
                split_tensor = torch.split(first_batch_element, 4, dim=0)[0]  # [4, 32, 32]
                processed_tensors.append(split_tensor)
            image = torch.stack(processed_tensors)

            from diffusers.models import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(f"pretrain/sd-vae-ft-ema").to(device)
            samples = vae.decode(image.to(device) / 0.18215).sample

            plot_path = f'./plot/282_pred/imgs/{images_num}'
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
            plot_path = f'./plot/282_pred/imgs/line/{images_num}'
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{i}.png")
            print(f"Plot saved to {plot_path}")
            plt.close()

        # block output
        if check_simularities == True:

            output_dirs_block = output_dirs + '/res_1'
            output_dir = f'{output_dirs_block}/{i}'
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

                # if flag == 2: 
                #     print(torch.sum(torch.abs(output_flat - final_output_flat), dim = 1))
                #     check = torch.abs(output_flat - final_output_flat)
                #     print(f"check:{check[0]}, output:{output_flat[0]}")
                #     cos_sim = cosine_similarity(output_flat, final_output_flat, dim=1)
                #     print(cos_sim)
                #     mean_cos_sim = cos_sim.mean().item()
                #     print(mean_cos_sim)
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
            plot_path = './plot/282_pred/line'
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

            plot_path = 'plot/282_pred/heatmap'
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{i}.png")
            print(f"Heatmap saved to {plot_path}")
            plt.close()

    # step_outputs = torch.cat(stepdirs)


if __name__ == "__main__":
    main()