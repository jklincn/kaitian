import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_mlu
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_dtype(torch.float32)


class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def generate_data(num_samples):
    images = torch.zeros((num_samples, 1, 128, 128))
    targets = torch.zeros_like(images)

    for i in range(num_samples):
        x, y = np.random.randint(0, 96, size=2)
        images[i, 0, x : x + 32, y : y + 32] = 1.0
        noise = torch.randn_like(images[i]) * 0.1
        images[i] += noise
        images[i] = torch.clamp(images[i], 0, 1)
        targets[i, 0, x : x + 32, y : y + 32] = 1.0

    return TensorDataset(images, targets)


dataset = generate_data(5000)
data_loader = DataLoader(dataset, batch_size=300, shuffle=True, num_workers=2)

device = "mlu"
model = DenoiseCNN().to(device)
optimizer = optim.Adam(model.parameters())
model.train()
criterion = nn.MSELoss()

start_time = time.time()
for _ in range(10):
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
end_time = time.time()

print(f"{(end_time - start_time):.2f}")
