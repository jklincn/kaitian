import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch_mlu
import torch_kaitian
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models

num_epochs = 1
lr = 0.001
batch_size = 32
device = torch_kaitian.device()


def setup(rank, size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group("kaitian", rank=rank, world_size=size)
    torch_kaitian.set_device(rank)


def run(rank, world_size):
    setup(rank, world_size)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2
    )

    model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                if rank == 0:
                    print(
                        f"Rank {rank}, Epoch {epoch+1}/{num_epochs}, Iteration {i}, Loss: {loss.item():.4f}"
                    )

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch_kaitian.init()
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
    )
