import torch
from torch import nn
from functools import partial

# Wrapper function for weight quantization
# Continous number of channel_group channels share the same quantization setup
@torch.no_grad()
def quantize_tensor_channel_group(W: torch.tensor, n_bits, group_size, tiling, sym, channel_group=1, clip_ratio=1.0, exponential=False, quant_type="int", quant_method="max") -> torch.tensor:
    assert W.is_contiguous(), "Input tensor is not contiguous"
    assert n_bits < 16

    if group_size > 0:
        assert W.shape[-1] % group_size == 0

    # group_size = 0 is per-channel quantization.
    if group_size == 0:
        W = quantize_tensor(W, n_bits=n_bits, group_size=0, tiling=tiling, sym=sym, exponential=exponential,quant_method=quant_method)
    else:
        for i1 in range(0, W.shape[1], group_size):
            i2 = min(i1 + group_size, W.shape[1])
            w = W[:,i1:i2]

            # Continous channels share the same quantization setup.
            # This trick is used for efficiency consideration.
            if channel_group > 1:
                w = w.reshape(int(W.shape[0]/channel_group), -1).contiguous() # Continous for bitsandbytes kernel calling

            # group_size is set to 0 because the layout is
            # already [num_groups, group_size]
            w = quantize_tensor(
                w,
                n_bits=n_bits,
                group_size=0,
                tiling=tiling,
                sym=sym,
                clip_ratio=clip_ratio,
                exponential=exponential,
                quant_type=quant_type,
                quant_method=quant_method
            )

            # Reshape back to original shape.
            if channel_group > 1:
                w = w.reshape(-1, group_size)
            W[:,i1:i2] = w

    return W.contiguous()


@torch.no_grad()
def quantize_tensor(w: torch.tensor, n_bits, group_size, tiling, sym, clip_ratio=1.0, exponential=False, quant_type="int", quant_method="max") -> torch.tensor:
    savedShape = w.shape
    w = w.squeeze()
    if not w.is_contiguous():
        w = w.contiguous()
    assert w.is_contiguous(), "tensor should be continous for bitsandbytes kernel."

    if tiling > 0:
        assert False, "16x16 Block-wise Quantization is abandoned in this project."

    if group_size > 0:
        assert w.shape[-1] % group_size == 0
        w = w.reshape(-1, group_size) # row-major order

    assert w.dim() == 2, "Weight format should be: [num_groups, group_size]"
    assert n_bits < 16

    def lp_loss(pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        y = torch.flatten(x, 1)
        return y.mean(1,keepdim=True)
    
    
    assert quant_type == "int", "Options should be in [int, fp]"
    if quant_method == "max":
        if sym:
            w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        else:
            w_max = w.amax(dim=-1, keepdim=True)
            w_min = w.amin(dim=-1, keepdim=True)

        if sym:
            q_max = (2**(n_bits-1)-1)
            q_min = (-2**(n_bits-1))
            if clip_ratio < 1.0:
                w_max = w_max * clip_ratio
            scales = w_max / q_max
            base = torch.zeros_like(scales)
        else:
            q_max = (2**(n_bits)-1)
            q_min = (0)
            if clip_ratio < 1.0:
                w_max *= clip_ratio
                w_min *= clip_ratio
            scales = (w_max-w_min).clamp(min=1e-5) / q_max
            base = torch.round(-w_min/scales).clamp_(min=q_min, max=q_max)
        w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
    
    elif quant_method == "mse":
        w_max = w.amax(dim=-1, keepdim=True)
        w_min = w.amin(dim=-1, keepdim=True)
        w_absmax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        best_score = torch.zeros_like(w_max) + (1e10)
        best_min = w_min.clone()
        best_max = w_max.clone()
        best_absmax = w_absmax.clone()
        for i in range(100):
            if sym:
                new_max = w_absmax * (1.0 - (i * 0.001))
                q_max = (2**(n_bits-1)-1)
                q_min = (-2**(n_bits-1))
                scales = new_max / q_max
                base = torch.zeros_like(scales)
            else:
                new_max = w_max * (1.0 - (i * 0.001))
                new_min = w_min * (1.0 - (i * 0.001))
                q_max = (2**(n_bits)-1)
                q_min = (0)
                scales = (new_max-new_min).clamp(min=1e-5) / q_max
                base = torch.round(-new_min/scales).clamp_(min=q_min, max=q_max)
            w_q = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales
            # L_p norm minimization as described in LAPQ
            # https://arxiv.org/abs/1911.07190
            score = lp_loss(w, w_q, p=2.4)
            if sym:
                best_absmax = torch.where(score < best_score, new_max, best_absmax)
            else:
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(best_score, score)
        # print('clip_ratio:', (best_absmax/w_absmax)) 
        if sym: 
            q_max = (2**(n_bits-1)-1)
            q_min = (-2**(n_bits-1))
            scales = best_absmax / q_max
            base = torch.zeros_like(scales)
        else:
            q_max = (2**(n_bits)-1)
            q_min = (0)
            scales = (best_max-best_min).clamp(min=1e-5) / q_max
            base = torch.round(-best_min/scales).clamp_(min=q_min, max=q_max)
        w = (torch.clamp(torch.round(w / scales) + base, q_min, q_max) - base) * scales

    else:
        raise NotImplementedError
    
    return w.reshape(savedShape)

# Wrapper function for activation quantization
# Simulate mixed-precision by decomposing input
@torch.no_grad()
def quantize_activation_wrapper(x: torch.tensor, args) -> torch.tensor:
    if args.abits >= 16:
        return x 
    
    qFunction = partial(
        quantize_tensor, 
        n_bits=args.abits, 
        group_size=args.act_group_size, 
        tiling=args.tiling, 
        sym=args.a_sym,
        clip_ratio=args.a_clip_ratio,
        exponential=False,
        quant_type=args.quant_type
    )

    savedShape = x.shape
    x = x.reshape(-1, savedShape[-1])
    assert args.act_group_size == 0 or (savedShape[-1]) % args.act_group_size == 0
    
    x = qFunction(x)

    return x.view(savedShape)

@torch.no_grad()
def quantize_attn_v_wrapper(w: torch.tensor, args) -> torch.tensor:
    # Input shape: [bsz, self.num_heads, seq_len, self.head_dim]
    # Quantize on head_dim
    assert w.shape[-1] == 72
    
    head_dim = w.shape[-1]
    saved_shape = w.shape
    w = w.reshape(-1, head_dim)

    w = quantize_tensor(w, n_bits=args.abits, group_size=0, tiling=0, sym=False, clip_ratio=args.kv_clip_ratio, exponential=False)
    return w.view(saved_shape)

@torch.no_grad()
def quantize_attn_k_wrapper(w: torch.tensor, args) -> torch.tensor:
    # Quantize on head_dim
    assert w.shape[-1] == 72
    
    head_dim = w.shape[-1]
    saved_shape = w.shape
    w = w.reshape(-1, head_dim)

    w = quantize_tensor(w, n_bits=args.abits, group_size=0, tiling=0, sym=False, clip_ratio=args.kv_clip_ratio, exponential=False)
    return w.view(saved_shape)

@torch.no_grad()
def quantize_attn_q_wrapper(w: torch.tensor, args) -> torch.tensor:
    # Quantize on head_dim
    assert w.shape[-1] == 72
    
    head_dim = w.shape[-1]
    saved_shape = w.shape
    w = w.reshape(-1, head_dim)

    w = quantize_tensor(w, n_bits=args.abits, group_size=0, tiling=0, sym=False, clip_ratio=args.kv_clip_ratio, exponential=False)
    return w.view(saved_shape)

class Quantizer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.register_buffer("scales", None)
        self.args = args
        # act_quant are configured outside.
        self.act_quant = lambda x: x

    @torch.no_grad()
    def forward(self, hidden_states):
        if self.args.static == False or self.scales is None:
            return self.act_quant(hidden_states)
        
        savedShape = hidden_states.shape
        assert self.scales is not None, "Scales is None"
        assert self.args.a_sym == False

        hidden_states = hidden_states.view(-1, savedShape[-1])
        selected_states = hidden_states.clone()

        if self.args.act_group_size > 0:
            selected_states = selected_states.reshape(-1, self.args.act_group_size)

        B, N, C = savedShape
        if self.args.act_group_size > 0:
            scales, base = self.scales[0].repeat(B * N, 1), self.scales[1].repeat(B * N, 1)
        else:
            scales, base = self.scales[0].unsqueeze(0).repeat(B * N, 1), self.scales[1].unsqueeze(0).repeat(B * N, 1)
        assert scales.numel() == selected_states.shape[-2], "Scales and selected states must have the same dimension"
        selected_states = (torch.clamp(torch.round(selected_states / scales) + base, self.q_min, self.q_max) - base) * scales
        selected_states = selected_states.reshape(-1, savedShape[-1])
        hidden_states = selected_states

        return hidden_states.view(savedShape)
    
    def to(self, *args, **kwargs):
        super(Quantizer, self).to(*args, **kwargs)
        if self.scales is not None:
            self.scales = self.scales.to(*args, **kwargs)
        return self

    def configure(self, func, scales):
        if self.args.static == False:
            self.act_quant = func
            return
        assert scales is not None, "Scales is None"
        self.register_buffer("scales", scales)
        if self.args.a_sym:
            self.q_min = (-2**(self.args.abits-1))
            self.q_max = (2**(self.args.abits-1)-1)
        else:
            self.q_min = (0)
            self.q_max = (2**(self.args.abits)-1)

    def extra_repr(self):
        return f'wbit={self.args.abits}, sym={self.args.a_sym}, group_size={self.args.act_group_size}, static={self.args.static}'