# KaiTian

> English | [简体中文](README_CN.md)

KaiTian is a Pytorch backend extension that enables distributed data parallel for heterogeneous devices.

The name comes from the theme song of the 2018 LPL Spring Finals, which also conveys the meaning of "creating the world". See https://www.youtube.com/watch?v=mGAjqYyVvzc 。

> Currently, only single-node multi-GPU DDP is supported. Multi-node multi-GPU DDP and model parallelism have not yet been implemented.

## Installation

### Prerequisites

- Python >= 3.8

- [Docker Engine](https://docs.docker.com/engine/install/)（Recommended >= 26.0.2）

- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) (optional)

  - [NVIDIA Driver](https://www.nvidia.com/Download/Find.aspx)（Recommended >= 520.61.05）

  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（Recommended >= 1.15.0）

- [Cambricon MLU](https://www.cambricon.com/) (optional)

  - [Cambricon MLU Driver](https://sdk.cambricon.com/download?component_name=Driver)（Recommended >= 5.10.22）

### Install KaiTian

```
git clone --recurse-submodules https://github.com/jklincn/kaitian.git
cd kaitian
python -m pip install -r requirements.txt
python -m pip install .
```

## Usage

### Adapt to KaiTian

Taking NVIDIA CUDA for distributed training as an example.

1. Import kaitian

   ```python
   import torch_kaitian
   from torch_kaitian.distributed import DistributedSampler, optimize_batch_size
   ```

2. Modify world_size

   ```python
   # Before
   world_size = torch.cuda.device_count()
   # After
   world_size = torch_kaitian.local_device_count()
   ```

3. Modify distributed setup

   ```python
   # Before
   dist.init_process_group("nccl", rank=rank, world_size=world_size)
   torch.cuda.set_device(rank)
   # After
   dist.init_process_group("kaitian", rank=rank, world_size=world_size)
   torch_kaitian.set_device(rank)
   ```

4. Modify device

   ```python
   # Before
   device = "cuda"
   # After
   device = torch_kaitian.device()
   ```

5. Modify DistributedSampler of **train_set** 

   ```python
   # Before
   train_sampler = DistributedSampler(train_set)
   # After
   train_sampler = DistributedSampler(train_set, batch_size)
   ```

6. Modify DataLoader of **train_set** 

   ```python
   # Before
   train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
   # After
   train_loader = DataLoader(train_set, batch_size=optimize_batch_size(batch_size), sampler=train_sampler, num_workers=2)
   ```

7. Modify random seed settings

   ```python
   # Before
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.backends.cudnn.deterministic = True
   # After
   torch_kaitian.manual_seed(seed)
   ```

Specific adaptation examples can be found in [example/cuda.py](example/cuda.py) (original code) and [example/kaitian.py](example/kaitian.py) (modified code)

### Initialize

Initialize the KaiTian environment, which includes creating a configuration file, registering available devices, and pulling images.

```
kaitian init
```

When there are changes in the machine environment (such as changes in the number of accelerators installed), reinitialization is necessary.

```
kaitian init -r
```

### Run

```
kaitian run -f your_code.py
```

By default, all available devices on the host will be used. You can specify which accelerators to use with `USE_XXX`. Currently supported options include `USE_CUDA` and `USE_MLU`.

Usage example:

- `USE_CUDA=0`：Use GPU0 to accelerate
- `USE_CUDA=0,1`：Use GPU0 and GPU1 to accelerate
- `USE_CUDA=0 USE_MLU=1`：Use GPU0 and MLU1 to accelerate
- `USE_CUDA=-1`：Don't use GPU to accelerate

## Image List

#### NVIDIA CUDA

- `0.0.0-cuda`
  - Python 3.10 + PyTorch 1.13.1 + CUDA 11.6 + cuDNN 8
  - FROM `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel`

#### Cambricon MLU

- `0.0.0-mlu`
  - Python 3.10 + Pytorch 1.13.1 + Cambricon 1.17.0
  - FROM `yellow.hub.cambricon.com/pytorch/pytorch:v1.17.0-torch1.13.1-ubuntu20.04-py310`