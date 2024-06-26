import torch
import torch.nn as nn
import functools
import math
from tqdm import tqdm
from qdit.qBlock import QLinearLayer

@torch.no_grad()
def get_act_stats(model, dataloader, device_, metric='hessian'):
    nsamples = len(dataloader)
    device = device_
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()

        if metric == 'hessian':
            tensorH = math.sqrt(2 / nsamples) * tensor.float().t()
            comming_H = tensorH.matmul(tensorH.t())
            comming_scales = torch.diag(comming_H)
        else:
            # Here we use abs since symmetric quantization use absmax.
            comming_scales = torch.mean(tensor.abs(), dim=0).float().cpu()

        if name in act_scales:
            if metric == 'hessian':
                act_scales[name] += comming_scales
            else:
                act_scales[name] = torch.max(act_scales[name], comming_scales)
        else:
            act_scales[name] = comming_scales

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.Tensor)
        if isinstance(y, tuple):
            y = y[0]
            assert isinstance(y, torch.Tensor)
        stat_tensor(name + ".input", x)
        stat_tensor(name + ".output", y)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )

    model.to(device)
    for calib_x, calib_t, calib_y in tqdm(dataloader):
        model(calib_x.to(device), calib_t.to(device), calib_y.to(device))

    for h in hooks:
        h.remove()

    return act_scales 


@torch.no_grad()
def get_act_scales(model, diffusion, dataloader, device_, args):
    device = device_
    act_scales = {}
    act_max_scales = {}
    act_min_scales = {}
    group_size = args.act_group_size[0]
    abits = args.abits

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        if group_size > 0:
            tensor = tensor.reshape(-1, hidden_dim//group_size, group_size)
        else:
            tensor = tensor.reshape(-1, hidden_dim)
        comming_max_scales = tensor.amax(dim=-1, keepdim=True)
        comming_max_scales, _ = torch.max(comming_max_scales, dim=0) # shape: (hidden_dim/group_size, 1) or (1)
        comming_min_scales = tensor.amin(dim=-1, keepdim=True)
        comming_min_scales, _ = torch.min(comming_min_scales, dim=0) # shape: (hidden_dim/group_size, 1) or (1)
        if name in act_max_scales:
            act_max_scales[name] = 0.9 * act_max_scales[name] + 0.1 * comming_max_scales
            act_min_scales[name] = 0.9 * act_min_scales[name] + 0.1 * comming_min_scales
        else:
            act_max_scales[name] = comming_max_scales
            act_min_scales[name] = comming_min_scales

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.Tensor)
    
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )

    model.to(device)

    for calib_x, calib_t, calib_y in tqdm(dataloader):
        model(calib_x.to(device), calib_t.to(device), calib_y.to(device))

    for h in hooks:
        h.remove()

    
    for name, value in act_max_scales.items():
        max_value = act_max_scales[name]
        min_value = act_min_scales[name]
        q_max = (2**(abits)-1)
        q_min = (0)
        scales = (max_value-min_value).clamp(min=1e-5) / q_max
        base = torch.round(-min_value/scales).clamp_(min=q_min, max=q_max)
        act_scales[name] = torch.stack([scales, base])
        

    return act_scales 