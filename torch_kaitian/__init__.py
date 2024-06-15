import os
import torch
import torch.distributed as dist

from . import _C

device_type = os.environ.get("DEVICE", None)
if device_type is None:
    raise EnvironmentError("Required environment variable DEVICE not set.")
elif device_type == "MLU":
    import torch_mlu


def device():
    return torch.device("privateuseone:0")


def set_device(rank):
    if device_type == "MLU":
        torch.mlu.set_device(rank)
    else:
        torch.cuda.set_device(rank)


def world_size():
    pass


def local_device_count():
    if device_type == "MLU":
        return torch.mlu.device_count()
    else:
        return torch.cuda.device_count()


def global_device_count():
    global_device_count_ = os.environ.get("KAITIAN_GLOBAL_DEVICE_COUNT", None)
    if global_device_count_ is None:
        raise EnvironmentError(
            f"Environment variable KAITIAN_GLOBAL_DEVICE_COUNT is not set."
        )
    else:
        return global_device_count_


def manual_seed(seed):
    torch.manual_seed(seed)
    if device_type == "MLU":
        torch.mlu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def time_spend():
    _C.time_spend()
