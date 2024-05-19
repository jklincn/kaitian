# KaiTian

KaiTian（开天）是 Pytorch 的通信后端扩展，它统一了各种集体通信库，为不同厂商加速卡的分布式训练提供了可能

名字源于2018年LPL春季总决赛的主题曲，详见 https://www.youtube.com/watch?v=mGAjqYyVvzc 。

## 硬件支持

### Cambricon MLU

#### 安装前提

需要预先安装：

- [Cambricon MLU Driver](https://sdk.cambricon.com/download?component_name=Driver)
- [Cambricon Neuware SDK](https://sdk.cambricon.com/download?component_name=Neuware+SDK)
- [Cambricon Pytorch](https://sdk.cambricon.com/download?component_name=PyTorch)

Cambricon Pytorch 的源码安装过程（包括 Cambricon Neuware SDK 安装）可以参考 [jklincn/cambricon-pytorch](https://github.com/jklincn/cambricon-pytorch) 。

#### 可配置环境变量

- NEUWARE_HOME：指向 Neuware SDK 的安装目录。默认值为 `/usr/local/neuware` 。

## 安装

当前通过测试的 pytorch 版本：

- [1.13.1](https://github.com/pytorch/pytorch/tree/v1.13.1)

```
python setup.py install
```

## 测试

```
# Only one linear layer and random data
python test/example_simple.py

# MobileNet_V2 and CIFAR10
python test/example.py
```

## 使用

导入 kaitian 前需要导入 torch 包，即

```
import torch
# import torch_mlu  # 如果使用 Cambricon MLU, 请取消注释
import torch_kaitian
```

以 cuda 和 nccl 为例

| 原始代码                                                     | 修改后代码                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| world_size = torch.cuda.device_count()                       | world_size = torch_kaitian.init()                            |
| backend = "nccl"<br />dist.init_process_group(backend, rank=rank, world_size=size) | backend = "kaitian"<br />dist.init_process_group(backend, rank=rank, world_size=size) |
| torch.cuda.set_device(rank)                                  | torch_kaitian.set_device(rank)                               |
| device = "cuda"<br />model = model.to(device)                | device = torch_kaitian.device()<br />model = model.to(device) |

