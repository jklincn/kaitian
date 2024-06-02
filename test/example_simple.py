import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch_mlu
import torch_kaitian
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

input_size = 1
output_size = 1
num_epochs = 100
num_samples = 100

device = torch_kaitian.device()


def setup(rank, size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group("kaitian", rank=rank, world_size=size)
    torch_kaitian.set_device(rank)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def train(rank, world_size):
    setup(rank, world_size)

    x = torch.randn(num_samples, input_size)
    y = 3 * x + 2 + torch.randn(num_samples, output_size) * 0.1
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)
    model = LinearRegressionModel().to(device)
    model = DDP(model, device_ids=[rank])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Rank {rank}, Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch_kaitian.init()
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
    )
