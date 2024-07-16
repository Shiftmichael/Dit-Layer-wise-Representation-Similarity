import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from train import center_crop_arr
import argparse

def generate_labels(label, count):
    return [label] * count

def main(args, label):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000


    # dataloader for class images
    data_dir = f'./nc/{label}_images'
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    for imgs, _ in dataloader:
        inv_normalize = transforms.Normalize(
            mean=[-1, -1, -1],
            std=[2, 2, 2]
        )
        imgs_inv = inv_normalize(imgs)
        save_dir = './results'
        save_image_path = os.path.join(save_dir, f'image_{label}.png')
        save_image(imgs_inv, save_image_path, nrow=10)
        break

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"pretrain/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    class_labels = generate_labels(label, 50)

    # Create sampling noise:
    n = 50
    # noise = torch.randn(n, 4, latent_size, latent_size, device=device)
    # z = vae.encode(imgs.to(device)).latent_dist.sample().mul_(0.18215)
    # t = torch.full((z.shape[0],), 249, device=device)
    # z = diffusion.q_sample(z, t, noise=noise)
    # y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    # y_null = torch.tensor([1000] * n, device=device)
    # y = torch.cat([y, y_null], 0)
    # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    # samples = diffusion.p_sample_loop(
    #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )
    # samples = diffusion.ddim_sample_loop(
    #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )
    model_pred = model.forward_with_cfg

    for input_t in range(0, 251, 10):
        if input_t == 250:
            input_t -= 1

        noise = torch.randn(n, 4, latent_size, latent_size, device=device)
        z = vae.encode(imgs.to(device)).latent_dist.sample().mul_(0.18215)
        t = torch.full((z.shape[0],), input_t, device=device)
        z = diffusion.q_sample(z, t, noise=noise)
        y = torch.tensor(class_labels, device=device)
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        new_t = torch.full((z.shape[0],), input_t, device=device)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        model_output = model_pred(z, new_t*4, **model_kwargs)
        B, C = z.shape[:2]
        model_output, _ = torch.split(model_output, C, dim=1)
        pred_xstart = diffusion._predict_xstart_from_eps(x_t=z, t=new_t, eps=model_output)
        model_mean, _, _ = diffusion.q_posterior_mean_variance(x_start=pred_xstart, x_t=z, t=new_t)
        eps, _ = model_mean.chunk(2, dim = 0)
        eps = vae.decode(eps / 0.18215).sample
        output_dir = f"./results//nc/{label}/pred_eps"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'sample_{label}_{input_t}.png')
        save_image(eps, output_path, nrow=5, normalize=True, value_range=(-1, 1))
        xstarts, _ = pred_xstart.chunk(2, dim = 0)
        samples = vae.decode(xstarts / 0.18215).sample
        output_dir = f"./results/nc/{label}/pred_xstart"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'sample_{label}_{input_t}.png')
        save_image(samples, output_path, nrow=5, normalize=True, value_range=(-1, 1))


    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    # output_dir = "./results"
    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, f'sample_{label}.png')
    # save_image(samples, output_path, nrow=5, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1120)
    parser.add_argument("--ckpt", type=str, default="pretrain/DiT-XL-2-256x256.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279, 112, 65]
    class_labels = [282, 207, 88, 417, 112]
    for label in class_labels:
        # args.seed = args.seed + 1
        main(args, label)
