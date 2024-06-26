"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import copy
import yaml
import logging
import time
from collections import Counter
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm
import math

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.download import find_model
from models.models import DiT_models
from utils.logger_setup import create_logger
from glob import glob
from copy import deepcopy

from qdit.quant import *
from qdit.outlier import *
from qdit.datautils import *
from collections import defaultdict
from qdit.modelutils import quantize_model, quantize_model_gptq,  add_act_quant_wrapper


from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

def validate_model(args, model, diffusion, vae):
    seed_everything(args.seed)
    device = next(model.parameters()).device
    using_cfg = args.cfg_scale > 1.0
    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, model.input_size, model.input_size, device=device)
    y = torch.tensor(class_labels, device=device)
    # Setup classifier-free guidance:
    if using_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        # sample_fn = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        # sample_fn = model.forward
    z = z.half()
    with autocast():
        samples = diffusion.ddim_sample_loop(
                model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    # Save and display images:
    save_image(samples, f'{args.experiment_dir}/sample.png', nrow=4, normalize=True, value_range=(-1, 1))
    print("Finish validating samples!")
    
def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def sample_fid(args, model, diffusion, vae):
    # Create folder to save samples:
    seed_everything(args.seed)
    device = next(model.parameters()).device
    using_cfg = args.cfg_scale > 1.0
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.seed}"
    sample_folder_dir = f"{args.experiment_dir}/{folder_name}"
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.batch_size
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / n) * n)
    print(f"Total number of images that will be sampled: {total_samples}")
    iterations = int(total_samples // n)
    pbar = range(iterations)
    pbar = tqdm(pbar)
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, model.input_size, model.input_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)

        z = z.half()
        with autocast():
            samples = diffusion.ddim_sample_loop(
                model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += n

    create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
    print("Done.")


def main():
    args = create_argparser().parse_args()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device: {device}')
    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    quant_method = "qdit"
    quant_string_name = f"{quant_method}_w{args.wbits}a{args.abits}"
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{quant_string_name}"  # Create an experiment folder
    args.experiment_dir = experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    create_logger(experiment_dir)
    logging.info(f"Experiment directory created at {experiment_dir}")
    logging.info(f"""wbits: {args.wbits}, abits: {args.abits}, w_sym: {args.w_sym}, a_sym: {args.a_sym},
                 weight_group_size: {args.weight_group_size}, act_group_size: {args.act_group_size},
                 quant_method: {args.quant_method}, use_gptq: {args.use_gptq}, static: {args.static},
                 image_size: {args.image_size}, cfg_scale: {args.cfg_scale}""")
    
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
    vae = AutoencoderKL.from_pretrained(f"../sd-vae-ft-{args.vae}").to(device)
    args.weight_group_size = eval(args.weight_group_size)
    args.act_group_size = eval(args.act_group_size)
    args.weight_group_size = [args.weight_group_size] * len(model.blocks) if isinstance(args.weight_group_size, int) else args.weight_group_size
    args.act_group_size = [args.act_group_size] * len(model.blocks) if isinstance(args.act_group_size, int) else args.act_group_size
    
    print("Inserting activations quantizers ...")
    if args.static:
        dataloader = get_loader(args.calib_data_path, nsamples=1024, batch_size=16)
        print("Getting activation stats...")
        scales = get_act_scales(
            model, diffusion, dataloader, device, args
        )
    else:
        scales = defaultdict(lambda: None)
    model = add_act_quant_wrapper(model, device=device, args=args, scales=scales)

    print("Quantizing ...")
    if args.use_gptq:
        dataloader = get_loader(args.calib_data_path, nsamples=256)
        model = quantize_model_gptq(model, device=device, args=args, dataloader=dataloader)
    else:
        model = quantize_model(model, device=device, args=args)

    print("Finish quant!")
    logging.info(model)

    # generate some sample images
    model.to(device)
    model.half()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    torch.set_grad_enabled(False)
    validate_model(args, model, diffusion, vae)
    
    # sample_fid(args, model, diffusion, vae)

def create_argparser():
    parser = argparse.ArgumentParser()

    # quantization parameters
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 5, 6, 8, 16],
        help='#bits to use for quantizing weight; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--abits', type=int, default=16, choices=[2, 3, 4, 5, 6, 8, 16],
        help='#bits to use for quantizing activation; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--exponential', action='store_true',
        help='Whether to use exponent-only for weight quantization.'
    )
    parser.add_argument(
        '--quantize_bmm_input', action='store_true',
        help='Whether to perform bmm input activation quantization. Default is not.'
    )
    parser.add_argument(
        '--a_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--w_sym', action='store_true',
        help='Whether to perform symmetric quantization. Default is asymmetric.'
    )
    parser.add_argument(
        '--static', action='store_true',
        help='Whether to perform static quantization (For activtions). Default is dynamic. (Deprecated in Atom)'
    )
    parser.add_argument(
        '--weight_group_size', type=str,
        help='Group size when quantizing weights. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--weight_channel_group', type=int, default=1,
        help='Group size of channels that will quantize together. (only for weights now)'
    )
    parser.add_argument(
        '--act_group_size', type=str,
        help='Group size when quantizing activations. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--tiling', type=int, default=0, choices=[0, 16],
        help='Tile-wise quantization granularity (Deprecated in Atom).'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--use_gptq', action='store_true',
        help='Whether to use GPTQ for weight quantization.'
    )
    parser.add_argument(
        '--quant_method', type=str, default='max', choices=['max', 'mse'],
        help='The method to quantize weight.'
    )
    parser.add_argument(
        '--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--w_clip_ratio', type=float, default=1.0,
        help='Clip ratio for weight quantization. new_max = max * clip_ratio'
    )
    parser.add_argument(
        '--save_dir', type=str, default='../saved',
        help='Path to store the reordering indices and quantized weights.'
    )
    parser.add_argument(
        '--quant_type', type=str, default='int', choices=['int', 'fp'],
        help='Determine the mapped data format by quant_type + n_bits. e.g. int8, fp4.'
    )
    parser.add_argument(
        '--calib_data_path', type=str, default='../cali_data/cali_data_4000.pth',
        help='Path to store the reordering indices and quantized weights.'
    )
    # Inherited from DiT
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--results-dir", type=str, default="../results")
    parser.add_argument(
        "--save_ckpt", action="store_true", help="choose to save the qnn checkpoint"
    )
    # sample_ddp.py
    parser.add_argument("--tf32", action="store_true",
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    return parser


if __name__ == "__main__": 
    main()
