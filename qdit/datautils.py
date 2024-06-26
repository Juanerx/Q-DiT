import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytorch_lightning import seed_everything
import torch.nn.functional as F


class CalibDataset(Dataset):
    def __init__(self, cali_data_path):
        data = torch.load(cali_data_path, map_location='cpu') ## its a list of tuples of tensors
        assert(len(data) == 3)
        self.xt = []
        self.t = []
        self.y = []
        nr_samples = data[0].shape[0]
        for i in range(nr_samples):
            self.xt.append(data[0][i])
            self.t.append(data[1][i])
            self.y.append(data[2][i])

    def __len__(self):
        return len(self.xt)
    
    def __getitem__(self, idx):
        return self.xt[idx], self.t[idx], self.y[idx]

class CalibDataset_t(Dataset):
    def __init__(self, cali_data_path, t):
        data = torch.load(cali_data_path, map_location='cpu') ## its a list of tuples of tensors
        assert(len(data) == 3)
        self.xt = []
        self.t = []
        self.y = []
        nr_samples = data[0].shape[0]
        for i in range(nr_samples):
            if data[1][i] == t:
                self.xt.append(data[0][i])
                self.t.append(data[1][i])
                self.y.append(data[2][i])

    def __len__(self):
        return len(self.xt)
    
    def __getitem__(self, idx):
        return self.xt[idx], self.t[idx], self.y[idx]

def get_loader(dataset_path, nsamples=1024, batch_size=32):
    seed_everything(42)
    dataset = CalibDataset(dataset_path)
    all_indices = list(range(len(dataset)))
    np.random.shuffle(all_indices)
    subset_indices = all_indices[:nsamples]
    sampler = SubsetRandomSampler(subset_indices)
    return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)

def get_loader_t(dataset_path, t, nsamples=1024, batch_size=32):
    seed_everything(42)
    dataset = CalibDataset_t(dataset_path, t)
    all_indices = list(range(len(dataset)))
    np.random.shuffle(all_indices)
    subset_indices = all_indices[:nsamples]
    sampler = SubsetRandomSampler(subset_indices)
    return DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)


def save_grad_data(model, qnn, block, dataloader):
    """
    Save gradient data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param block: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param damping: damping the second-order gradient by adding some constant in the FIM diagonal
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: gradient data
    """
    device = next(model.parameters()).device
    get_grad = GetBlockGrad(model, qnn, device)
    cached_batches = []
    torch.cuda.empty_cache()

    for batch_data in dataloader:
        cur_grad = get_grad(batch_data, block)
        cached_batches.append(cur_grad.cpu())

    cached_grads = torch.cat([x for x in cached_batches])
    cached_grads = cached_grads.abs().pow(2)
    # scaling to make sure its mean is 1
    # cached_grads = cached_grads * torch.sqrt(cached_grads.numel() / cached_grads.pow(2).sum())
    torch.cuda.empty_cache()
    return cached_grads


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetBlockGrad:
    def __init__(self, model, qnn, device: torch.device):
        self.model = model
        self.qnn = qnn
        self.device = device
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input, block):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()
        print(block)
        handle = block.register_full_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.qnn.zero_grad()
                x, t, y = model_input[0].to(self.device), model_input[1].to(self.device), model_input[2].to(self.device)
                out_fp = self.model(x, t, y)
                # quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = self.qnn(x, t, y)
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                print(loss)
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        return self.data_saver.grad_out.data
