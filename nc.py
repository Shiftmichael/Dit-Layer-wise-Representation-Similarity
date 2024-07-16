import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import cosine_similarity
import seaborn as sns
from torchvision.utils import save_image

from metric.nc import compute_nc1_per_layer
from metric.cl_dl import compute_cl_dl, compute_cl_dl_return_all, compute_cl
from metric.unif import lunif
from metric.cdnv import compute_cdnv



@torch.no_grad()
def main():
    steps = 250
    check_simularities = True
    labels = [282, 207, 88, 417, 112]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dirs = 'block_output/nc_max'
    save_dirs = 'simularity/nc'
    for index in range(steps, -1, -10):
        if index == 250:
            i = index - 1
        else:
            i = index 

        print(index)

        # block output
        if check_simularities == True:
            classes_mean_layers = []
            classes_layers = []
            classes_mean_layers_head0 = []
            classes_layers_head0 = []
            for label in labels:    
                output_dir = f'{output_dirs}/{label}/res_1/{i}'
                saved_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
                saved_files.sort()

                outputs = []
                for file_name in saved_files:
                    file_path = os.path.join(output_dir, file_name)
                    output = torch.load(file_path).to(device)
                    outputs.append(output)
                
                # for j, output in enumerate(outputs):
                #     print(f"Output {j} shape: {output.shape}")
                    
                final_output = outputs[-1]
            
                class_mean_layers = []
                class_layers = []
                class_mean_layers_head0 = []
                class_layers_head0 = []
                flag = 0
                for output in outputs:
                    flag += 1
                    
                    # output, _ = output.chunk(2, dim = 0) # Remove null class samples
                    num_tensors = output.shape[0]
                    output_flat = output.reshape(output.size(0), -1)
                    output_flat /= torch.norm(output_flat, p=2)
                    output_flat_head0 = output[:, 0]
                    mean_layer = output_flat.mean(dim=0, keepdim=True)
                    mean_layer_head0 = output_flat_head0.mean(dim=0, keepdim=True)
                    class_mean_layers.append(mean_layer)
                    class_layers.append(output_flat)
                    class_mean_layers_head0.append(mean_layer_head0)
                    class_layers_head0.append(output_flat_head0)

                class_mean_layers = torch.cat(class_mean_layers)
                class_layers = torch.stack(class_layers)
                class_mean_layers_head0 = torch.cat(class_mean_layers_head0)
                class_layers_head0 = torch.stack(class_layers_head0)

                classes_mean_layers.append(class_mean_layers)
                classes_layers.append(class_layers)
                classes_mean_layers_head0.append(class_mean_layers_head0)
                classes_layers_head0.append(class_layers_head0)

            
            classes_mean_layers = torch.stack(classes_mean_layers)
            classes_layers = torch.stack(classes_layers)
            classes_mean_layers_head0 = torch.stack(classes_mean_layers_head0)
            classes_layers_head0 = torch.stack(classes_layers_head0)

            # nc1 = compute_nc1_per_layer(classes_layers_head0, classes_mean_layers_head0)
            cl, dl = compute_cl_dl(classes_layers, classes_mean_layers)
            cl_head0 = compute_cl(classes_layers_head0)
            # _, _, Sw, Sb, g_mean = compute_cl_dl_return_all(classes_layers, classes_mean_layers)
            # print(f'SigmaW 0: {Sw[0]}, 5: {Sw[5]}, 11: {Sw[11]}, 17: {Sw[17]}, 27: {Sw[27]},')
            # print(f'SigmaB 0: {Sb[0]}, 5: {Sb[5]}, 11: {Sb[11]}, 17: {Sb[17]}, 27: {Sb[27]},')
            # print(f'SigmaB 0: {g_mean[0]}, 5: {g_mean[5]}, 11: {g_mean[11]}, 17: {g_mean[17]}, 27: {g_mean[27]},')
            # unif = lunif(classes_layers)
            cdnv = compute_cdnv(classes_layers)
            cdnv_head0 = compute_cdnv(classes_layers_head0)
            
            # nc1 = nc1.to('cpu').numpy()
            # layers = np.arange(len(nc1))
            # plt.figure(figsize=(10, 8))
            # plt.plot(layers, nc1, marker='o', label='nc1')
            # plt.title('Neural Collapse 1')
            # plt.xlabel('Layers')
            # plt.ylabel('NC1')
            # plot_path = './plot/nc_max/nc1'
            # os.makedirs(plot_path, exist_ok=True)
            # plt.savefig(f"{plot_path}/{i}.png")
            # print(f"Plot saved to {plot_path}")
            # plt.close()

            cl = cl.to('cpu').numpy()
            layers = np.arange(len(cl))
            plt.figure(figsize=(10, 8))
            plt.plot(layers, cl, marker='o', label='cl')
            plt.yscale('log')
            plt.title('Cl')
            plt.xlabel('Layers')
            plt.ylabel('Cl')
            plot_path = './plot/nc_max/cl'
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{i}.png")
            # print(f"Plot saved to {plot_path}")
            plt.close()

            cl_head0 = cl_head0.to('cpu').numpy()
            layers = np.arange(len(cl))
            plt.figure(figsize=(10, 8))
            plt.plot(layers, cl_head0, marker='o', label='cl')
            plt.yscale('log')
            plt.title('Cl_0')
            plt.xlabel('Layers')
            plt.ylabel('Cl')
            plot_path = './plot/nc_max/cl_0'
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{i}.png")
            # print(f"Plot saved to {plot_path}")
            plt.close()

            dl = dl.to('cpu').numpy()
            layers = np.arange(len(dl))
            plt.figure(figsize=(10, 8))
            plt.plot(layers, dl, marker='o', label='dl')
            plt.yscale('log')
            plt.title('Dl')
            plt.xlabel('Layers')
            plt.ylabel('Dl')
            plot_path = './plot/nc_max/dl'
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{i}.png")
            # print(f"Plot saved to {plot_path}")
            plt.close()

            # unif = unif.to('cpu').numpy()
            # layers = np.arange(len(unif))
            # plt.figure(figsize=(10, 8))
            # plt.plot(layers, unif, marker='o', label='unif')
            # plt.title('Uniform')
            # plt.xlabel('Layers')
            # plt.ylabel('Uniform')
            # plot_path = './plot/nc_max/unif'
            # os.makedirs(plot_path, exist_ok=True)
            # plt.savefig(f"{plot_path}/{i}.png")
            # # print(f"Plot saved to {plot_path}")
            # plt.close()

            cdnv = cdnv.to('cpu').numpy()
            layers = np.arange(len(cdnv))
            plt.figure(figsize=(10, 8))
            plt.plot(layers, cdnv, marker='o', label='cdnv')
            plt.yscale('log')
            plt.title('CDNV')
            plt.xlabel('Layers')
            plt.ylabel('CDNV')
            plot_path = './plot/nc_max/CDNV'
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{i}.png")
            # print(f"Plot saved to {plot_path}")
            plt.close()

            cdnv_head0 = cdnv_head0.to('cpu').numpy()
            layers = np.arange(len(cdnv_head0))
            plt.figure(figsize=(10, 8))
            plt.plot(layers, cdnv_head0, marker='o', label='cdnv')
            plt.yscale('log')
            plt.title('CDNV')
            plt.xlabel('Layers')
            plt.ylabel('CDNV')
            plot_path = './plot/nc_max/CDNV_0'
            os.makedirs(plot_path, exist_ok=True)
            plt.savefig(f"{plot_path}/{i}.png")
            # print(f"Plot saved to {plot_path}")
            plt.close()


if __name__ == "__main__":
    main()