import os

import torch

device_type = os.environ.get("DEVICE", None)
if device_type:
    from . import _C

    if device_type == "MLU":
        import torch_mlu
    elif device_type == "CUDA":
        pass


def device():
    return torch.device("privateuseone:0")


def set_device(rank: int):
    if device_type == "MLU":
        torch.mlu.set_device(rank)
    else:
        torch.cuda.set_device(rank)


def local_device_count():
    if device_type == "MLU":
        return torch.mlu.device_count()
    else:
        return torch.cuda.device_count()


def manual_seed(seed):
    torch.manual_seed(seed)
    if device_type == "MLU":
        torch.mlu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def time_spend():
    _C.time_spend()
