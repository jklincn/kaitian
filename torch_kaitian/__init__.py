import os

import torch

from . import _C

device_type = os.environ.get("DEVICE", None)
if device_type is None:
    raise EnvironmentError("Required environment variable DEVICE not set.")
elif device_type == "MLU":
    import torch_mlu


def device():
    return torch.device("privateuseone:0")


def set_device(rank):
    if rank == 0:
        # Automatic init and finalize: https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html
        from mpi4py import MPI
    if device_type == "MLU":
        torch.mlu.set_device(rank)
    else:
        torch.cuda.set_device(rank)


def world_size():
    pass


def device_count():
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