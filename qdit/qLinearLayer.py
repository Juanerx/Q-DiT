import torch
import torch.nn as nn
from .quant import quantize_tensor, quantize_tensor_channel_group

def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        if module.enable_quant:
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        args,
        enable_quant: bool = True
    ):
        super().__init__()
        self.args = args
        self.register_buffer('weight', originalLayer.weight.data)
        self.enable_quant = enable_quant # whether to allow quant on weights, default True
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias.data)
        else:
            self.bias = None
        self.quantized = False
        
    @torch.no_grad()
    def forward(self, x):
        y = torch.functional.F.linear(x, self.weight, self.bias)
        return y
    
    def to(self, *args, **kwargs):
        super(QLinearLayer, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        return self
    
    @torch.no_grad()
    def quant(self):
        if self.args.wbits >= 16:
            return

        self.weight = quantize_tensor_channel_group(
            self.weight.clone(), 
            n_bits=self.args.wbits,
            exponential=self.args.exponential, 
            sym=self.args.w_sym,
            group_size=self.args.weight_group_size,
            channel_group=self.args.weight_channel_group,
            clip_ratio=self.args.w_clip_ratio,
            tiling=self.args.tiling,
            quant_type=self.args.quant_type,
            quant_method=self.args.quant_method,
        )

        self.quantized = True
        return
    
    def extra_repr(self):
        return f'wbit={self.args.wbits}, sym={self.args.w_sym}, group_size={self.args.weight_group_size}, channel_group={self.args.weight_channel_group}, quantized={self.quantized}'