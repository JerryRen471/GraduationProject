import sys
sys.path.insert(0, sys.path[0]+"/../")

import torch as tc
from torch import nn
from Library.DataFun import split_time_series

from matplotlib import pyplot as plt

class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()

def txt2tensor(filename):
    file = open(filename, mode='r')
    lines = file.readlines()
    file.close()
    for _ in range(len(lines)):
        lines[_] = float(lines[_][:-1])
    data = tc.tensor(lines)
    return data

sample = txt2tensor('D:\\Manual\\Code\\tests\\Graduation Project\\data.txt')

model = RNN(1, 100, 4)
model = tc.load('D:\\Manual\\Code\\tests\\net.pth')

dataset, labels = split_time_series(sample, length=8, device=tc.device('cuda:0'), dtype=tc.float64)
output = model(dataset.reshape(dataset.shape + (1, ))).data

x = tc.tensor([_ for _ in range(sample.shape[0])])
legends = []
plt.plot(x, sample, label='磁矩真实的时间演化')
plt.plot(x, tc.cat([sample[:8].to(device=tc.device('cpu')), output.to(device=tc.device('cpu'))], dim=0), label='利用LSTM预测的磁矩随时间演化')
xlabel = 't'
ylabel = 'magnetic moment'
title = '利用初态为|0...01>的磁矩演化数据训练得到的模型预测初态为|10...0>的磁矩演化'
plt.legend()
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.show()