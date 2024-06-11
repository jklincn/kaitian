import torch
import importlib.util

from . import _C

is_mlu = importlib.util.find_spec("torch_mlu") is not None

if is_mlu:
    import torch_mlu


def device():
    return torch.device("privateuseone:0")


def set_device(rank):
    return _C.set_device(rank)


def world_size():
    return _C.world_size()


def device_count():
    if is_mlu:
        return torch.mlu.device_count()
    else:
        return torch.cuda.device_count()


def manual_seed(seed):
    torch.manual_seed(seed)
    if is_mlu:
        torch.mlu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
