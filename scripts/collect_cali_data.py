# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.download import find_model
from models.models import DiT_models
import argparse
import math
from tqdm import tqdm
import numpy as np
import os

cali_data_dir = "../cali_data"

def sample_cali_data_per_batch(input_list_per_batch, batch_size=32):
    """
    x: (N, C, H, W), t: (N,), y: (N,).
    """
    timesteps = len(input_list_per_batch)
    cali_data = []
    unique_t = np.random.choice(range(0, timesteps), batch_size, replace=False)
    for idx in range(batch_size):
        t = unique_t[idx]
        samples_t = input_list_per_batch[t]
        cali_data.append([samples_t[0][idx], samples_t[1][idx], samples_t[2][idx]])
            
    return [torch.stack([sample[i] for sample in cali_data]) for i in range(len(cali_data[0]))]

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        record_inputs=True
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"../sd-vae-ft-{args.vae}").to(device)
    using_cfg = args.cfg_scale > 1.0

    
    iterations = int(math.ceil(args.num_cali_data / args.batch_size))
    pbar = range(iterations)
    pbar = tqdm(pbar)
    cali_data = []
    for batch_idx in pbar:
        # Sample inputs:
        z = torch.randn(args.batch_size, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * args.batch_size, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)
        samples = diffusion.ddim_sample_loop(
            model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        input_list = model.get_input_list()
        cali_data_per_batch = sample_cali_data_per_batch(input_list, batch_size=args.batch_size)
        cali_data.append(cali_data_per_batch)
        model.reset_input_list()
    
    cali_data = [torch.cat([batch[i] for batch in cali_data]) for i in range(len(cali_data[0]))]
    cali_data = [data[:args.num_cali_data] for data in cali_data]

    if not os.path.exists(cali_data_dir):
        os.mkdir(cali_data_dir)

    filename = f"cali_data_{args.image_size}.pth"
    torch.save(cali_data, os.path.join(cali_data_dir, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-cali-data", type=int, default=256)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
