import math
import os
from typing import Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.distributed import Dataset
from torch.utils.data.distributed import Sampler

T_co = TypeVar("T_co", covariant=True)

device_type = os.environ.get("DEVICE", None)
if device_type:
    from . import _C

    if device_type == "MLU":
        import torch_mlu
    elif device_type == "CUDA":
        pass


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


# We don't choose batch sizes as powers of 2
# reference:
# https://sebastianraschka.com/blog/2022/batch-size-2.html
# https://wandb.ai/datenzauberai/Batch-Size-Testing/reports/Do-Batch-Sizes-Actually-Need-To-Be-Powers-of-2---VmlldzoyMDkwNDQx


def set_batch_size(ori_batch_size):
    pass


class DistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        num_replicas = global_world_size()
        rank = global_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        print("len: ", len(self.dataset))
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
