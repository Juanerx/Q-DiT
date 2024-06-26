import gc
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from qdit.qLinearLayer import find_qlinear_layers
from qdit.qBlock import QuantDiTBlock
from qdit.gptq import GPTQ, Quantizer_GPTQ
from functools import partial
from models.models import DiTBlock

from .quant import quantize_activation_wrapper, quantize_attn_v_wrapper, quantize_attn_k_wrapper, quantize_attn_q_wrapper


def add_act_quant_wrapper(model, device, args, scales):
    blocks = model.blocks
    
    for i in range(len(blocks)):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]
        m = None
        if isinstance(blocks[i], DiTBlock):
            m = QuantDiTBlock(
                dit_block=blocks[i],
                args=args_i,
            )
        elif isinstance(blocks[i], QuantDiTBlock):
            m = blocks[i]

        if m is None:
            continue

        m = m.to(device)

        nameTemplate = 'blocks.{}.{}.{}'
        m.attn.input_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'attn', 'qkv')]
        )
        m.attn.act_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'attn', 'proj')]
        )
        if args.quantize_bmm_input:
            m.attn.q_quant.configure(
                partial(quantize_attn_q_wrapper, args=args_i),
                None
            )
            m.attn.k_quant.configure(
                partial(quantize_attn_k_wrapper, args=args_i),
                None
            )
            m.attn.v_quant.configure(
                partial(quantize_attn_v_wrapper, args=args_i),
                None
            )

        m.mlp.input_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'mlp', 'fc1')]
        )
        m.mlp.act_quant.configure(
            partial(quantize_activation_wrapper, args=args_i),
            scales[nameTemplate.format(i, 'mlp', 'fc2')]
        )
        
        
        blocks[i] = m.cpu()
        torch.cuda.empty_cache()
    return model

def quantize_model(model, device, args):
    blocks = model.blocks
    for i in tqdm(range(len(blocks))):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]
        m = None
        if isinstance(blocks[i], DiTBlock):
            m = QuantDiTBlock(
                dit_block=blocks[i],
                args=args_i,
            )
        elif isinstance(blocks[i], QuantDiTBlock):
            m = blocks[i]

        if m is None:
            continue

        m = m.to(device)
        m.mlp.fc1.quant()
        m.mlp.fc2.quant()
        m.attn.qkv.quant()
        m.attn.proj.quant()

        blocks[i] = m.cpu()
        torch.cuda.empty_cache()
    return model

def quantize_layer(model, name, device):
    blocks = model.blocks
    i = int(name.split(".")[1])
    assert(isinstance(blocks[i], QuantDiTBlock))
    m = blocks[i]
    m = m.to(device)

    if name.endswith("mlp.fc1"):
        m.mlp.fc1.quant()
    elif name.endswith("mlp.fc2"):
        m.mlp.fc2.quant()
    elif name.endswith("attn.qkv"):
        m.attn.qkv.quant()
    elif name.endswith("attn.proj"):
        m.attn.proj.quant()
    else:
        raise NotImplementedError

    blocks[i] = m.cpu()
    torch.cuda.empty_cache()
    return model

def quantize_block(block, device):
    assert(isinstance(block, QuantDiTBlock))
    block.to(device)

    block.mlp.fc1.quant()
    block.mlp.fc2.quant()
    block.attn.qkv.quant()
    block.attn.proj.quant()

    torch.cuda.empty_cache()

def quantize_model_gptq(model, device, args, dataloader):
    print('Starting GPTQ quantization ...')
    blocks = model.blocks
    
    quantizers = {}
    for i in tqdm(range(len(blocks))):
        args_i = copy.deepcopy(args)
        args_i.weight_group_size = args.weight_group_size[i]
        args_i.act_group_size = args.act_group_size[i]
        if isinstance(blocks[i], DiTBlock):
            m = QuantDiTBlock(
                dit_block=blocks[i],
                args=args_i,
            )
        elif isinstance(blocks[i], QuantDiTBlock):
            m = blocks[i]
        else:
            continue

        block = m.to(device)

        block_layers = find_qlinear_layers(block)

        sequential = [list(block_layers.keys())]
       
        for names in sequential:
            subset = {n: block_layers[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer_GPTQ()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.w_sym, mse=False, 
                    channel_group=args.weight_channel_group,
                    clip_ratio=args.w_clip_ratio,
                    quant_type=args.quant_type
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            model.to(device)
            for calib_x, calib_t, calib_y in tqdm(dataloader):
                model(calib_x.to(device), calib_t.to(device), calib_y.to(device))

            for h in handles:
                h.remove()
            
            for name in subset:
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.weight_group_size[0]
                )
                subset[name].quantized = True
                quantizers['model.blocks.%d.%s' % (i, name)] = gptq[name].quantizer.cpu()
                gptq[name].free()

            del gptq

        blocks[i] = block.cpu()
        del block, m
        torch.cuda.empty_cache()
        gc.collect()

    return model
