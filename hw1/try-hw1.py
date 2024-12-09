import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class COVID_Data(Dataset):
    def __init__(self):
        self.data = pd.read_csv('covid.train.csv')
        self.data = np.array(self.data)
        self.features = self.data[:, 41:-1]
        self.labels = self.data[:, -1]

        # 数据归一化（Standardization）
        self.features = (self.features - np.mean(self.features, axis=0)) / np.std(self.features, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feature = torch.tensor(self.features[index], dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.float)
        return feature, label


dataset = COVID_Data()
#之前我没shuffle
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(53, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.squeeze(1)  # 去除多余的维度
        return x


net = Net()
#adam
optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001)
loss_func = nn.MSELoss()

nums_epoch = 10
losses = []

for epoch in range(nums_epoch):
    print(f'第{epoch + 1}轮训练开始:')
    curEpoch_loss = 0
    for batch_idx, (features, labels) in enumerate(train_loader):
        pred = net(features)
        loss = loss_func(pred, labels)
        curEpoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:  # 每10个批次打印一次
            print(f'第{batch_idx}批次，损失{loss.item()}')

    losses.append(curEpoch_loss / len(train_loader))

# 绘制损失曲线
plt.plot(range(nums_epoch), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()