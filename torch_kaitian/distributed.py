import math
import os
from typing import Iterator, Optional, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import Dataset, Sampler


from .config import MAX_COMPUTE_CAPABILITY

CUDA_COMPUTE_CAPABILTY = float(os.environ.get("CUDA_COMPUTE_CAPABILTY", "6"))
T_co = TypeVar("T_co", covariant=True)


def global_rank() -> int:
    local_rank = dist.get_rank()
    global_rank_start = os.environ.get("KAITIAN_GLOBAL_RANK_START", None)
    if global_rank_start is None:
        raise EnvironmentError(
            "[KaiTian] [Internal Error] Required environment variable KAITIAN_GLOBAL_RANK_START not set."
        )
    return int(global_rank_start) + local_rank


def global_world_size() -> int:
    global_world_size_ = os.environ.get("KAITIAN_GLOBAL_WORLD_SIZE", None)
    if global_world_size_ is None:
        raise EnvironmentError(
            "[KaiTian] [Internal Error] Required environment variable KAITIAN_GLOBAL_WORLD_SIZE not set."
        )
    return int(global_world_size_)


# We don't choose batch sizes as powers of 2
# reference:
# https://sebastianraschka.com/blog/2022/batch-size-2.html
# https://wandb.ai/datenzauberai/Batch-Size-Testing/reports/Do-Batch-Sizes-Actually-Need-To-Be-Powers-of-2---VmlldzoyMDkwNDQx


def optimize_batch_size(ori_batch_size: int):
    rank = global_rank()
    compute_capability = get_compute_capability(rank)

    batch_size = round(ori_batch_size * compute_capability / MAX_COMPUTE_CAPABILITY)
    return batch_size


def compute_capability() -> float:
    compute_capability_ = os.environ.get("KAITIAN_COMPUTE_CAPABILTY", None)
    if compute_capability_ is None:
        raise EnvironmentError(
            "[KaiTian] [Internal Error] Required environment variable KAITIAN_COMPUTE_CAPABILTY not set."
        )
    return float(compute_capability_)


def total_compute_capability() -> float:
    total_compute_capability_ = os.environ.get("KAITIAN_TOTAL_COMPUTE_CAPABILTY", None)
    if total_compute_capability_ is None:
        raise EnvironmentError(
            "[KaiTian] [Internal Error] Required environment variable KAITIAN_TOTAL_COMPUTE_CAPABILTY not set."
        )
    return float(total_compute_capability_)


def get_compute_capability(global_rank: int) -> float:
    if global_rank == 0:
        return CUDA_COMPUTE_CAPABILTY
    elif global_rank == 1:
        return 10.0


def get_sampler_total_size(dataset_len: int) -> int:
    cuda_batch_size = round(64 * get_compute_capability(0) / MAX_COMPUTE_CAPABILITY)
    mlu_batch_size = 64
    iterations = math.ceil(dataset_len / (cuda_batch_size + mlu_batch_size))
    total_size = iterations * (cuda_batch_size + mlu_batch_size)
    return total_size


def get_num_samples(dataset_len: int, global_rank: int) -> int:
    cuda_batch_size = round(64 * get_compute_capability(0) / MAX_COMPUTE_CAPABILITY)
    mlu_batch_size = 64
    iterations = math.ceil(dataset_len / (cuda_batch_size + mlu_batch_size))
    if global_rank == 0:
        return iterations * cuda_batch_size
    elif global_rank == 1:
        return iterations * mlu_batch_size


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
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        # drop_last always is False
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
        self.batch_size = optimize_batch_size(batch_size)
        self.num_samples = get_num_samples(len(dataset), rank)

        print(
            f"rank: {rank}, batch_size: {self.batch_size}, num_samples: {self.num_samples}"
        )

        self.total_size = get_sampler_total_size(len(self.dataset))
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        assert len(indices) == self.total_size

        # subsample
        start_index = 0
        for i in range(global_world_size()):
            if i == self.rank:
                break
            start_index += get_num_samples(len(self.dataset), i)

        end_index = start_index + self.num_samples
        indices = indices[start_index:end_index]
        print(
            f"rank: {self.rank}, start_index: {start_index}, end_index:{end_index}, len_indices:{len(indices)}",
        )
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
