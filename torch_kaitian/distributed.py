import math
import os
from typing import Iterator, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.distributed import Dataset, Sampler

from . import config, redis

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


def calc_optimized_batch_size(global_rank: int, ori_batch_size: int) -> int:
    compute_capability = redis.get_compute_capability(global_rank)
    # We don't choose batch sizes as powers of 2
    # reference:
    # https://sebastianraschka.com/blog/2022/batch-size-2.html
    # https://wandb.ai/datenzauberai/Batch-Size-Testing/reports/Do-Batch-Sizes-Actually-Need-To-Be-Powers-of-2---VmlldzoyMDkwNDQx
    batch_size = round(
        ori_batch_size * compute_capability / config.MAX_COMPUTE_CAPABILITY
    )
    return batch_size


def optimize_batch_size(ori_batch_size: int) -> int:
    rank = global_rank()
    return calc_optimized_batch_size(rank, ori_batch_size)


def get_total_optimized_batch_size(ori_batch_size: int) -> int:
    total_optimized_batch_size = 0
    world_size = global_world_size()
    for rank in range(world_size):
        total_optimized_batch_size += calc_optimized_batch_size(rank, ori_batch_size)
    return total_optimized_batch_size


def get_num_samples(dataset_len: int, global_rank: int, ori_batch_size: int) -> int:
    iterations = calc_iteration_times(dataset_len, ori_batch_size)
    optimized_batch_size = calc_optimized_batch_size(global_rank, ori_batch_size)
    return iterations * optimized_batch_size


def calc_iteration_times(dataset_len: int, ori_batch_size: int) -> int:
    total_batch_size = get_total_optimized_batch_size(ori_batch_size)
    iterations = math.ceil(dataset_len / total_batch_size)
    return iterations


class DistributedSampler(Sampler[T_co]):
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
        dataset_len = len(dataset)
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.global_rank = rank
        self.epoch = 0
        self.batch_size = batch_size
        self.optimized_batch_size = calc_optimized_batch_size(rank, batch_size)
        self.iterations = calc_iteration_times(dataset_len, batch_size)
        self.num_samples = get_num_samples(dataset_len, self.global_rank, batch_size)
        self.total_size = self.iterations * get_total_optimized_batch_size(batch_size)
        self.shuffle = shuffle
        self.seed = seed

        # print(
        #     f"global_rank: {self.global_rank}, optimized_batch_size: {self.optimized_batch_size}, num_samples: {self.num_samples}, iterations:{self.iterations}, total_size: {self.total_size}"
        # )

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
            if i == self.global_rank:
                break
            start_index += get_num_samples(len(self.dataset), i, self.batch_size)
        end_index = start_index + self.num_samples
        indices = indices[start_index:end_index]

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
