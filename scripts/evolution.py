# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import sys
import time
from PIL import Image
from pytorch_lightning import seed_everything
import math
import argparse
import os
import logging

import torchvision.transforms as transforms
import collections
sys.setrecursionlimit(10000)
import functools

import argparse
import os

from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.download import find_model
from models.models import DiT_models
from utils.logger_setup import create_logger
from glob import glob
from copy import deepcopy

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import autocast

from qdit.quant import *
from qdit.outlier import *
from qdit.datautils import *
from collections import defaultdict
from qdit.modelutils import quantize_model,  add_act_quant_wrapper
from qdit.qBlock import QuantDiTBlock
from qdit.datautils import get_loader
from evaluations.evaluator import Evaluator
import tensorflow.compat.v1 as tf


print = functools.partial(print, flush=True)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

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

class EvolutionSearcher(object):

    def __init__(self, args, model, diffusion, device, search_space=None, efficiency_predictor=None, evaluator=None):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.evaluator = evaluator
        ## EA hyperparameters
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.constraint = args.constraint
        ## tracking variable 
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {} # {str(cand): {'visited':, 'loss':,}
        self.search_space = search_space
        self.efficiency_predictor = efficiency_predictor
        ## sampling parameters
        self.criterion = nn.MSELoss()
        self.z = None
        self.model_kwargs = None
        self.samples_fp = None
        self.interm_samples_fp = []
        self.dataloader = get_loader(args.calib_data_path, nsamples=1024, batch_size=32)


    def sample_subnet(self):
        subnet = random.choices(self.search_space, k=112)
        return subnet

    def is_legal(self, cand):
        if self.efficiency_predictor(cand) < self.constraint:
            return False 
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        info['loss'] = self.get_cand_loss(args=self.args, cand=eval(cand), device=self.device)
        logging.info('cand: {}, loss: {}'.format(cand, info['loss']))

        info['visited'] = True
        return True

    def configure_group_size(self, model, group_size):
        blocks = model.blocks
        assert(len(group_size) == len(blocks)*4)
        for i in range(len(blocks)):
            assert(isinstance(blocks[i], QuantDiTBlock))
            m = blocks[i]
            m.attn.input_quant.args.act_group_size = group_size[i*4]
            m.attn.qkv.args.weight_group_size = group_size[i*4]

            m.attn.act_quant.args.act_group_size = group_size[i*4+1]
            m.attn.proj.args.weight_group_size = group_size[i*4+1]

            m.mlp.input_quant.args.act_group_size = group_size[i*4+2]
            m.mlp.fc1.args.weight_group_size = group_size[i*4+2]

            m.mlp.act_quant.args.act_group_size = group_size[i*4+3]
            m.mlp.fc2.args.weight_group_size = group_size[i*4+3]
            torch.cuda.empty_cache()
        return model

    def get_cand_loss(self, cand, args, device):
        # FID appraoch
        seed_everything(args.seed)
        qnn = deepcopy(self.model)
        scales = defaultdict(lambda: None)
        qnn = add_act_quant_wrapper(qnn, device=device, args=args, scales=scales)
        self.configure_group_size(qnn, cand)
        qnn = quantize_model(qnn, device, args)
        int_image_dir = f'{args.experiment_dir}/int_samples'
        os.makedirs(int_image_dir, exist_ok=True)
        decoded_samples = None
        qnn.to(device)
        n = self.args.batch_size
        total_samples = int(math.ceil(self.args.num_fid_samples / n) * n)
        iterations = int(total_samples // n)
        pbar = tqdm(range(iterations))
        total = 0
        for _ in pbar:
            # Sample inputs:
            z = torch.randn(n, self.model.in_channels, self.model.input_size, self.model.input_size, device=self.device)
            y = torch.randint(0, self.args.num_classes, (n,), device=self.device)

            # Setup classifier-free guidance:
            if self.args.cfg_scale:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            else:
                model_kwargs = dict(y=y)

            z = z.half()
            with autocast():
                samples = diffusion.ddim_sample_loop(
                    qnn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )
            if self.args.cfg_scale:
                samples, _ = samples.chunk(2, dim=0)

            samples = vae.decode(samples / 0.18215).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for i, sample in enumerate(samples):
                index = i + total
                Image.fromarray(sample).save(f"{int_image_dir}/{index:06d}.png")
            total += n
        sample_batch = create_npz_from_sample_folder(int_image_dir, args.num_fid_samples)

        ref_acts = self.evaluator.read_activations(args.ref_batch)
        ref_stats, ref_stats_spatial = evaluator.read_statistics(args.ref_batch, ref_acts)

        sample_acts = self.evaluator.read_activations(sample_batch)
        sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_batch, sample_acts)

        fid_score = sample_stats.frechet_distance(ref_stats)
        loss = fid_score

        return loss
    
    def get_random(self, num, preset=None):
        logging.info('random select ........')
        if preset:
            for cand in preset:
                cand = str(cand)
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
                info['loss'] = self.get_cand_loss(args=self.args, cand=eval(cand), device=self.device)
                logging.info('cand: {}, loss: {}'.format(cand, info['loss']))
                info['visited'] = True
                self.candidates.append(cand)
                logging.info('random {}/{}'.format(len(self.candidates), num))
        while len(self.candidates) < num:
            cand = self.sample_subnet()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))

    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logging.info('cross ......')
        res = []
        max_iters = cross_num * 10

        def random_cross():
            cand1 = choice(self.keep_top_k[k])
            cand2 = choice(self.keep_top_k[k])

            new_cand = []
            cand1 = eval(cand1)
            cand2 = eval(cand2)

            for i in range(len(cand1)):
                if np.random.random_sample() < 0.5:
                    new_cand.append(cand1[i])
                else:
                    new_cand.append(cand2[i])

            return new_cand

        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            cand = random_cross()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('cross {}/{}'.format(len(res), cross_num))

        logging.info('cross_num = {}'.format(len(res)))
        return res

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)
            for i in range(len(cand)):
                if random.random() < m_prob:
                    cand[i] = random.choice(self.search_space)
            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def search(self):
        logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

        preset = []
        preset.append([128]*112)
        self.get_random(self.population_num, preset=preset)

        while self.epoch < self.max_epochs:
            logging.info('epoch = {}'.format(self.epoch))
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['loss'])
            logging.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[self.select_num])))
            for i, cand in enumerate(self.keep_top_k[self.select_num]):
                logging.info('No.{} {} loss = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['loss']))
            
            # sys.exit()
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)

            self.candidates = mutation

            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand

            self.get_random(self.population_num) #变异+杂交凑不足population size的部分重新随机采样

            self.epoch += 1

def efficiency_predictor(cand):
    cand = eval(cand)
    return sum(cand) / len(cand)

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_batch", type=str, default='', help="path to reference batch npz file")
    # Evolutionary search parameters
    parser.add_argument(
        '--max_epochs', type=int, default=20,
        help='Max number of epochs for search.'
    )
    parser.add_argument(
        '--select_num', type=int, default=10,
        help='Top k select number.'
    )
    parser.add_argument(
        '--population_num', type=int, default=50,
    )
    parser.add_argument(
        '--crossover_num', type=int, default=15,
    )
    parser.add_argument(
        '--mutation_num', type=int, default=25,
    )
    parser.add_argument(
        '--m_prob', type=float, default=0.1,
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5,
    )
    parser.add_argument(
        '--constraint', type=int, default=128,
    )
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
        '--weight_group_size', type=int, default=0, choices=[0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96, 128, 144, 192, 288, 384, 576, 1152],
        help='Group size when quantizing weights. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--weight_channel_group', type=int, default=1,
        help='Group size of channels that will quantize together. (only for weights now)'
    )
    parser.add_argument(
        '--act_group_size', type=int, default=0, choices=[0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96, 128, 144, 192, 288, 384, 576, 1152],
        help='Group size when quantizing activations. Using 128 as default quantization group.'
    )
    parser.add_argument(
        '--tiling', type=int, default=0, choices=[0, 16],
        help='Tile-wise quantization granularity (Deprecated in Atom).'
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
        '--calib_data_path', type=str, default='../cali_data/cali_data_256.pth',
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
    # atom
    return parser

if __name__ == '__main__':

    args = create_argparser().parse_args()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
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
                 weight_group_size: {args.weight_group_size}, act_group_size: {args.act_group_size}
                 quant_method: {args.quant_method}, use_gptq: {args.use_gptq}""")
    
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
    args.weight_group_size = [args.weight_group_size] * len(model.blocks)
    args.act_group_size = [args.act_group_size] * len(model.blocks)
    search_space = [32, 64, 128, 192, 288]
    ## build EA 
    config = tf.ConfigProto(
        allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))
    evaluator.warmup()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)
    t = time.time()
    searcher = EvolutionSearcher(args, model=model, diffusion=diffusion, search_space=search_space, device=device, efficiency_predictor=efficiency_predictor, evaluator=evaluator)
    searcher.search()
    logging.info('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))
