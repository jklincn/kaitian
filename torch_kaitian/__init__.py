import os
import torch
import torch.distributed as dist

from . import _C

device_type = os.environ.get("DEVICE", None)
if device_type is None:
    raise EnvironmentError(
        "[KaiTian] [Internal Error] Required environment variable DEVICE not set."
    )
elif device_type == "MLU":
    import torch_mlu


def device():
    return torch.device("privateuseone:0")


def global_rank():
    global_rank_ = os.environ.get("KAITIAN_GLOBAL_RANK_START", None)
    if global_rank_ is None:
        raise EnvironmentError(
            "[KaiTian] [Internal Error] Required environment variable KAITIAN_GLOBAL_RANK_START not set."
        )
    return int(global_rank_) + dist.get_rank()


def set_device(rank):
    if device_type == "MLU":
        torch.mlu.set_device(rank)
    else:
        torch.cuda.set_device(rank)


def global_world_size():
    global_world_size_ = os.environ.get("KAITIAN_GLOBAL_WORLD_SIZE", None)
    if global_world_size_ is None:
        raise EnvironmentError(
            "[KaiTian] [Internal Error] Required environment variable KAITIAN_GLOBAL_WORLD_SIZE not set."
        )
    return int(global_world_size_)


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
