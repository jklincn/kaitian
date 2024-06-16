# KaiTian

> English | [简体中文](README_CN.md)

KaiTian is a Pytorch backend extension that enables distributed data parallel for heterogeneous devices.

The name comes from the theme song of the 2018 LPL Spring Finals, which also conveys the meaning of "creating the world". See https://www.youtube.com/watch?v=mGAjqYyVvzc 。

> Currently, only single-node multi-GPU DDP is supported. Multi-node multi-GPU DDP and model parallelism have not yet been implemented.

## Installation

### Prerequisites

#### Basic Environment

- Python >= 3.8
- [Docker Engine](https://docs.docker.com/engine/install/)（Recommended >= 26.0.2）

#### NVIDIA CUDA Support

- [NVIDIA Driver](https://www.nvidia.com/Download/Find.aspx)（Recommended >= 520.61.05）
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（Recommended >= 1.15.0）

#### Cambricon MLU Support

- [Cambricon MLU Driver](https://sdk.cambricon.com/download?component_name=Driver)（Recommended >= 5.10.22）

### Install KaiTian

```
git clone --recurse-submodules https://github.com/jklincn/kaitian.git
cd kaitian
pip install -r requirements.txt
```

## Usage

### Adapt to KaiTian

Taking NVIDIA CUDA for distributed training as an example.

| Original code                                                | Modified code                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              | import torch_kaitian                                         |
| world_size = torch.cuda.device_count()                       | world_size = torch_kaitian.local_device_count()              |
| dist.init_process_group("nccl", rank=rank, world_size=world_size) | dist.init_process_group("kaitian", rank=rank, world_size=world_size) |
| torch.cuda.set_device(rank)                                  | torch_kaitian.set_device(rank)                               |
| device = "cuda"                                              | device = torch_kaitian.device()                              |
| DistributedSampler(train_set, num_replicas=world_size, rank=rank) | global_world_size = torch_kaitian.global_world_size()<br />global_rank = torch_kaitian.global_rank() <br />DistributedSampler(train_set,num_replicas=global_world_size,rank=global_rank ) |
| torch.manual_seed(seed)<br />torch.cuda.manual_seed(seed)<br />torch.backends.cudnn.deterministic = True | torch_kaitian.manual_seed(seed)                              |

Specific adaptation examples can be found in [example/cuda.py](example/cuda.py) (original code) and [example/kaitian.py](example/kaitian.py) (modified code), with the following differences：

```
~/kaitian/example$ diff cuda.py kaitian.py
14a15,16
> import torch_kaitian
> 
18c20
< device = "cuda"
---
> device = torch_kaitian.device()
24,25c26,27
<     dist.init_process_group("nccl", rank=rank, world_size=world_size)
<     torch.cuda.set_device(rank)
---
>     dist.init_process_group("kaitian", rank=rank, world_size=world_size)
>     torch_kaitian.set_device(rank)
29,31c31
<     torch.manual_seed(seed)
<     torch.cuda.manual_seed(seed)
<     torch.backends.cudnn.deterministic = True
---
>     torch_kaitian.manual_seed(seed)
45d44
< 
47c46,50
<     train_sampler = DistributedSampler(train_set)
---
>     train_sampler = DistributedSampler(
>         train_set,
>         num_replicas=torch_kaitian.global_world_size(),
>         rank=torch_kaitian.global_rank(),
>     )
53c56,61
<     test_sampler = DistributedSampler(test_set, shuffle=False)
---
>     test_sampler = DistributedSampler(
>         test_set,
>         num_replicas=torch_kaitian.global_world_size(),
>         rank=torch_kaitian.global_rank(),
>         shuffle=False,
>     )
97c105
<     world_size = torch.cuda.device_count()
---
>     world_size = torch_kaitian.local_device_count()
```

### Pull relevant images

```
docker pull jklincn/kaitian:[tag]
```

> For example, if you want to train using CUDA and MLU simultaneously, you need to pull the CUDA and MLU images.

#### NVIDIA CUDA

- `0.0.0-cuda`
  - Python 3.10 + PyTorch 1.13.1 + CUDA 11.6 + cuDNN 8
  - FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

#### Cambricon MLU

- `0.0.0-mlu`
  - Python 3.10 + Pytorch 1.13.1 + Cambricon 1.17.0
  - FROM yellow.hub.cambricon.com/pytorch/pytorch:v1.17.0-torch1.13.1-ubuntu20.04-py310

### Run code using launcher

```
python run.py your_code.py
```

By default, all available devices on the host will be used. Specific device can be disabled through `USE_XXX=0`. Currently, `USE_CUDA` and `USE_MLU` is supported.

