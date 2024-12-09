import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

class COVID_Dataset(Dataset):
    def __init__(self,data,flag):
        if flag=='test':
            #注意看那个测试集数据
            self.features=data[:,41:]
            self.labels=None
            self.features = (self.features - np.mean(self.features, axis=0)) / np.std(self.features, axis=0)
        else:
            self.features=data[:,41:-1]
            self.labels=data[:,-1]
            # # 数据归一化
            self.features = (self.features - np.mean(self.features, axis=0)) / np.std(self.features, axis=0)

    def __getitem__(self, item):
        if self.labels is None:
            #就是测试集
            features=torch.tensor(self.features[item],dtype=torch.float)
            return features
        else:
            features=torch.tensor(self.features[item],dtype=torch.float)
            labels=torch.tensor(self.labels[item],dtype=torch.float)
            return features,labels

    def __len__(self):
        return len(self.features)
#先把数据从csv读取出来
data=pd.read_csv('covid.train.csv')
data = data.values  # 将 DataFrame 转换为 numpy 数组
print(type(data))
train_data_len=int(len(data)*0.8)
valid_data_len=len(data)-train_data_len
#返回的是dataset对象?
train_data,valid_data=torch.utils.data.random_split(data,[train_data_len,valid_data_len])
print(type(train_data))

train_data=np.array(train_data)
valid_data=np.array(valid_data)
print(type(train_data))

train_set=COVID_Dataset(train_data,'train')
valid_set=COVID_Dataset(valid_data,'valid')

train_loader=DataLoader(train_set,batch_size=32,shuffle=True)
valid_loader=DataLoader(valid_set,batch_size=32,shuffle=False)


test_data=pd.read_csv('covid.test.csv')
test_data=test_data.values
print(type(test_data))

test_set=COVID_Dataset(test_data,'test')
test_loader=DataLoader(test_set,batch_size=32,shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=torch.nn.Sequential(
            nn.Linear(53,32),
            nn.ReLU(),
            nn.Linear(32,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )

    def forward(self,x):
        x=self.model(x)
        x = x.squeeze(1)#正则化之后loss没有异常的高了,然后再来这一下就好了
        return x

net=Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

def train():

    net.train()#是不是要有这个
    losses=[]
    for epoch in range(100):
        print('第{}轮训练开始了'.format(epoch))
        curEpochLoss=0
        for features,labels in train_loader:
            pred=net(features)
            #这块有点问题
            #labels从文件读取之后就是(32,)是一维的
            #pred经过forward()之后是(32,1)是二维的

            loss=loss_func(pred,labels)
            curEpochLoss+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        losses.append(curEpochLoss/len(train_loader))
        print('第{}轮,loss={}'.format(epoch,curEpochLoss/len(train_loader)))

    plt.plot(range(100),losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.show()

def valid():
    #TODO
    net.eval()
    losses=[]

    with torch.no_grad():#要有括号
        for features,labels in valid_loader:
            pred=net(features)
            loss=loss_func(pred,labels)
            losses.append(loss.item())
    #这里是在做验证,range()不是100了
    plt.plot(range(len(valid_loader)),losses)
    #多加点信息
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.title('validation loss')
    plt.show()

def test():
    net.eval()
    list=[]
    with torch.no_grad():
        for features in test_loader:
            preds=net(features)
            for pred in preds.detach():
                list.append(pred.item())

    # for temp in list:
    #     print(temp)
        # 转换为DataFrame并保存为CSV
    results = pd.DataFrame({
        'id': range(len(test_data)),
        'tested_positive': list
    })
    results.to_csv('test_predictions.csv', index=False)


train()
valid()
test()
