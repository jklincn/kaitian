# KaiTian

> [English](README.md) | 简体中文

KaiTian（开天）是 PyTorch 的通信后端扩展，实现了异构加速卡的分布式训练。

名字源于 2018 年 LPL 春季总决赛的主题曲，也为“开天辟地”之意，详见 https://www.bilibili.com/video/BV1jW411V78P 。

> 目前仅支持单机多卡 DDP，多机多卡 DDP 以及模型并行等暂未实现。

## 安装

### 前提

#### 基础环境

- Python >= 3.8
- [Docker Engine](https://docs.docker.com/engine/install/)（推荐 >= 26.0.2）

#### NVIDIA CUDA 支持

- [NVIDIA Driver](https://www.nvidia.com/Download/Find.aspx)（推荐 >= 520.61.05）
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（推荐 >= 1.15.0）

#### Cambricon MLU 支持

- [Cambricon MLU Driver](https://sdk.cambricon.com/download?component_name=Driver)（推荐 >= 5.10.22）

### 安装 KaiTian

```
git clone --recurse-submodules https://github.com/jklincn/kaitian.git
cd kaitian
python -m pip install -r requirements.txt
python -m pip install .
```

## 使用

### 适配 KaiTian 框架

以使用 NVIDIA CUDA 进行分布式训练为例

| 原始代码                                                     | 修改后代码                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              | import torch_kaitian                                         |
| world_size = torch.cuda.device_count()                       | world_size = torch_kaitian.local_device_count()              |
| dist.init_process_group("nccl", rank=rank, world_size=world_size) | dist.init_process_group("kaitian", rank=rank, world_size=world_size) |
| torch.cuda.set_device(rank)                                  | torch_kaitian.set_device(rank)                               |
| device = "cuda"                                              | device = torch_kaitian.device()                              |
| DistributedSampler(train_set, num_replicas=world_size, rank=rank) | global_world_size = torch_kaitian.global_world_size()<br />global_rank = torch_kaitian.global_rank() <br />DistributedSampler(train_set,num_replicas=global_world_size,rank=global_rank ) |
| torch.manual_seed(seed)<br />torch.cuda.manual_seed(seed)<br />torch.backends.cudnn.deterministic = True | torch_kaitian.manual_seed(seed)                              |

具体适配示例可见 [example/cuda.py](example/cuda.py)（原始代码）以及 [example/kaitian.py](example/kaitian.py)（修改后代码），差异如下：

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

### 拉取相关镜像

```
docker pull jklincn/kaitian:[tag]
```

> 比如想使用 CUDA 和 MLU 同时训练，则需要拉取 CUDA 和 MLU 两个镜像。

#### NVIDIA CUDA 

- `0.0.0-cuda`
  - Python 3.10 + PyTorch 1.13.1 + CUDA 11.6 + cuDNN 8
  - 基础镜像为 pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

#### Cambricon MLU

- `0.0.0-mlu`
  - Python 3.10 + Pytorch 1.13.1 + Cambricon 1.17.0
  - 基础镜像为 yellow.hub.cambricon.com/pytorch/pytorch:v1.17.0-torch1.13.1-ubuntu20.04-py310

### 使用启动器运行训练代码

```
kaitian run your_code.py
```

默认使用主机上所有可用的加速卡，可以通过 `USE_XXX=0` 禁用特定加速卡，目前支持 `USE_CUDA`、`USE_MLU`