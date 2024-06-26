import torch
from torch import nn
from typing import List, Optional, Tuple
import math
from .quant import Quantizer, quantize_tensor, quantize_tensor_channel_group
from .qLinearLayer import QLinearLayer
from models.models import DiTBlock, modulate, TimestepEmbedder
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import torch.nn.functional as F
from copy import deepcopy

class QuantDiTBlock(nn.Module):
    def __init__(
        self,
        dit_block: DiTBlock,
        args
    ):
        super().__init__()
        self.args = args
        # self.hidden_size = originalLayer.hidden_size
        self.quantize_bmm_input = args.quantize_bmm_input
        self.attn = QuantAttention(dit_block.attn, deepcopy(args))
        self.norm1 = dit_block.norm1
        self.mlp = QuantMlp(dit_block.mlp, deepcopy(args))
        self.norm2 = dit_block.norm2
        self.adaLN_modulation = nn.Sequential(
            dit_block.adaLN_modulation[0],
            QLinearLayer(dit_block.adaLN_modulation[1], deepcopy(args))
        )
        if self.quantize_bmm_input:
            self.ln1_quant = Quantizer(args=deepcopy(args))
            self.attn_quant = Quantizer(args=deepcopy(args))
            self.ln2_quant = Quantizer(args=deepcopy(args))
            self.mlp_quant = Quantizer(args=deepcopy(args))
            self.adaln_quant = Quantizer(args=deepcopy(args))

    def to(self, *args, **kwargs):
        super(QuantDiTBlock, self).to(*args, **kwargs)
        self.attn = self.attn.to(*args, **kwargs)
        self.mlp = self.mlp.to(*args, **kwargs)
        self.norm1 = self.norm1.to(*args, **kwargs)
        self.norm2 = self.norm2.to(*args, **kwargs)
        self.adaLN_modulation = self.adaLN_modulation.to(*args, **kwargs)
        if self.quantize_bmm_input:
            self.ln1_quant = self.ln1_quant.to(*args, **kwargs)
            self.attn_quant = self.attn_quant.to(*args, **kwargs)
            self.ln2_quant = self.ln2_quant.to(*args, **kwargs)
            self.mlp_quant = self.mlp_quant.to(*args, **kwargs)
            self.adaln_quant = self.adaln_quant.to(*args, **kwargs)
        return self

    def forward(self, x, c):
        if not self.quantize_bmm_input:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_quant(self.adaLN_modulation(c)).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn_quant(self.attn(modulate(self.ln1_quant(self.norm1(x)), shift_msa, scale_msa)))
            x = x + gate_mlp.unsqueeze(1) * self.mlp_quant(self.mlp(modulate(self.ln2_quant(self.norm2(x)), shift_mlp, scale_mlp)))
        return x


class QuantAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, 
        attn: Attention,
        args
    ):
        super().__init__()
        self.quantize_bmm_input = args.quantize_bmm_input
        self.abits = args.abits
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scale = attn.scale
        self.fused_attn = attn.fused_attn
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.attn_drop = attn.attn_drop
        self.proj_drop = attn.proj_drop
        
        self.input_quant = Quantizer(args=deepcopy(args))
        self.qkv = QLinearLayer(attn.qkv, deepcopy(args))
        if self.quantize_bmm_input:
            self.q_quant = Quantizer(args=deepcopy(args))
            self.k_quant = Quantizer(args=deepcopy(args))
            self.v_quant = Quantizer(args=deepcopy(args))
        self.act_quant = Quantizer(args=deepcopy(args))
        self.proj = QLinearLayer(attn.proj, deepcopy(args))
        self.register_buffer("reorder_index_qkv", None)
        self.register_buffer("reorder_index_proj", None)

    def to(self, *args, **kwargs):
        super(QuantAttention, self).to(*args, **kwargs)
        self.qkv = self.qkv.to(*args, **kwargs)
        self.proj = self.proj.to(*args, **kwargs)
        self.input_quant = self.input_quant.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        if self.quantize_bmm_input:
            self.q_quant = self.q_quant.to(*args, **kwargs)
            self.v_quant = self.v_quant.to(*args, **kwargs)
            self.k_quant = self.k_quant.to(*args, **kwargs)
        if self.reorder_index_qkv is not None:
            self.reorder_index_qkv = self.reorder_index_qkv.to(*args, **kwargs)
        if self.reorder_index_proj is not None:
            self.reorder_index_proj = self.reorder_index_proj.to(*args, **kwargs)
        return self

    def forward(self, x):
        B, N, C = x.shape
        if self.reorder_index_qkv is not None:
            x = torch.index_select(x, 2, self.reorder_index_qkv)
        x = self.input_quant(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.quantize_bmm_input:
            q = self.q_quant(q)
            k = self.k_quant(k)
            v = self.v_quant(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        if self.reorder_index_proj is not None:
            x = torch.index_select(x, 2, self.reorder_index_proj)
        x = self.act_quant(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class QuantMlp(nn.Module):
    def __init__(
        self,
        mlp: Mlp,
        args
    ):
        super().__init__()
        self.input_quant = Quantizer(args=deepcopy(args))
        self.fc1 = QLinearLayer(mlp.fc1, deepcopy(args))
        self.act = mlp.act
        self.drop1 = mlp.drop1
        self.norm = mlp.norm
        self.act_quant = Quantizer(args=deepcopy(args))
        self.fc2 = QLinearLayer(mlp.fc2, deepcopy(args))
        self.drop2 = mlp.drop2
        self.register_buffer("reorder_index_fc1", None)
        # self.register_buffer("act_shifts", None)

    def to(self, *args, **kwargs):
        super(QuantMlp, self).to(*args, **kwargs)
        self.fc1 = self.fc1.to(*args, **kwargs)
        self.act = self.act.to(*args, **kwargs)
        self.drop1 = self.drop1.to(*args, **kwargs)
        self.norm = self.norm.to(*args, **kwargs)
        self.fc2 = self.fc2.to(*args, **kwargs)
        self.drop2 = self.drop2.to(*args, **kwargs)
        self.act_quant = self.act_quant.to(*args, **kwargs)
        self.input_quant = self.input_quant.to(*args, **kwargs)
        if self.reorder_index_fc1 is not None:
            self.reorder_index_fc1 = self.reorder_index_fc1.to(*args, **kwargs)
        return self
    
    def forward(self, x):
        if self.reorder_index_fc1 is not None:
            x = torch.index_select(x, 2, self.reorder_index_fc1)
        x = self.input_quant(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.act_quant(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

