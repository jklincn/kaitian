# KaiTian

> [English](README.md) | 简体中文

KaiTian（开天）是 PyTorch 的通信后端扩展，实现了异构加速卡的分布式训练。

名字源于 2018 年 LPL 春季总决赛的主题曲，也为“开天辟地”之意，详见 https://www.bilibili.com/video/BV1jW411V78P 。

> 目前仅支持单机多卡 DDP，多机多卡 DDP 以及模型并行等暂未实现。

## 安装

### 前提

- Python >= 3.8

- [Docker Engine](https://docs.docker.com/engine/install/)（推荐 >= 26.0.2）

- NVIDIA CUDA（可选）

  - [NVIDIA Driver](https://www.nvidia.com/Download/Find.aspx)（推荐 >= 520.61.05）

  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（推荐 >= 1.15.0）

- Cambricon MLU（可选）

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

1. 导入 kaitian 包

   ```python
   import torch_kaitian
   from torch_kaitian.distributed import DistributedSampler, optimize_batch_size
   ```

2. 修改 world_size

   ```python
   # 修改前
   world_size = torch.cuda.device_count()
   # 修改后
   world_size = torch_kaitian.local_device_count()
   ```

3. 修改分布式设置

   ```python
   # 修改前
   dist.init_process_group("nccl", rank=rank, world_size=world_size)
   torch.cuda.set_device(rank)
   # 修改后
   dist.init_process_group("kaitian", rank=rank, world_size=world_size)
   torch_kaitian.set_device(rank)
   ```

4. 修改设备

   ```python
   # 修改前
   device = "cuda"
   # 修改后
   device = torch_kaitian.device()
   ```

5. 修改**训练数据集**的分布式采样器

   ```python
   # 修改前
   train_sampler = DistributedSampler(train_set)
   # 修改后
   train_sampler = DistributedSampler(train_set, batch_size)
   ```

6. 修改**训练数据集**的数据加载器

   ```python
   # 修改前
   train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
   # 修改后
   train_loader = DataLoader(train_set, batch_size=optimize_batch_size(batch_size), sampler=train_sampler, num_workers=2)
   ```

7. 修改随机种子设置

   ```python
   # 修改前
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.backends.cudnn.deterministic = True
   # 修改后
   torch_kaitian.manual_seed(seed)
   ```

具体适配示例可见 [example/cuda.py](example/cuda.py)（原始代码）以及 [example/kaitian.py](example/kaitian.py)（修改后代码）

### 初始化

```
kaitian init
```

使用以上命令进行运行环境初始化（自动化流程包括创建配置文件、注册可用设备、拉取镜像）

当机器环境有变化（比如安装的加速卡数量发生变动）时，需要重新进行初始化

```
kaitian init -r
```

### 运行

```
kaitian run -f your_code.py
```

默认使用主机上所有可用的加速卡，可以通过 `USE_XXX` 指定使用的加速卡，目前支持 `USE_CUDA`、`USE_MLU`

使用示例：

- `USE_CUDA=0`：使用 GPU0 进行加速
- `USE_CUDA=0,1`：使用 GPU0 和 GPU1 进行加速
- `USE_CUDA=0 USE_MLU=1`：使用 GPU0 和 MLU1 进行加速
- `USE_CUDA=-1`：不使用 GPU 进行加速

## 镜像列表

#### NVIDIA CUDA 

- `0.0.0-cuda`
  - Python 3.10 + PyTorch 1.13.1 + CUDA 11.6 + cuDNN 8
  - 基础镜像为 `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel`

#### Cambricon MLU

- `0.0.0-mlu`
  - Python 3.10 + Pytorch 1.13.1 + Cambricon 1.17.0
  - 基础镜像为 `yellow.hub.cambricon.com/pytorch/pytorch:v1.17.0-torch1.13.1-ubuntu20.04-py310`