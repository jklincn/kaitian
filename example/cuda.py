import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models

num_epochs = 1
lr = 0.001
batch_size = 64
device = "cuda"


def setup(rank, size):
    dist.init_process_group("nccl", rank=rank, world_size=size)
    torch.cuda.set_device(rank)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run(rank, world_size):
    setup(rank, world_size)
    set_seed(world_size)  # using a random number is also acceptable
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=transform
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=transform
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_set, num_replicas=world_size, rank=rank, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, sampler=test_sampler, num_workers=2
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
                        f"Rank {rank}, Epoch {epoch+1}/{num_epochs}, Iteration {i}, Loss: {loss.item():.4f}",
                        flush=True,
                    )
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if rank == 0:
        # Accuracy: 75.84%
        print(f"Accuracy: {100 * correct / total:.2f}%", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    # Download in advance to avoid duplicate downloads by multiple processes
    torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
    start_time = time.time()
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
    )
    end_time = time.time()
    print(f"Time spent: {(end_time - start_time):.6f} seconds")