import torch as tc
import numpy as np
from matplotlib import pyplot as plt

def txt2tensor(filename):
    file = open(filename, mode='r')
    lines = file.readlines()
    file.close()
    for _ in range(len(lines)):
        lines[_] = float(lines[_][:-1])
    data = tc.tensor(lines)
    return data

sample = txt2tensor('D:\\Manual\\Code\\tests\\Graduation Project\\data.txt')
train_pre = txt2tensor('D:\\Manual\\Code\\tests\\LSTMdata\\train_pre.txt')
train_loss = txt2tensor('D:\\Manual\\Code\\tests\\LSTMdata\\train_loss.txt')
test_pre = txt2tensor('D:\\Manual\\Code\\tests\\LSTMdata\\test_pre.txt')
test_loss = txt2tensor('D:\\Manual\\Code\\tests\\LSTMdata\\test_loss.txt')

# print(sample.shape)
# print(train_pre.shape)
# print(test_pre.shape)

x = tc.tensor([_ for _ in range(sample.shape[0])])
legends = []
plt.plot(x, sample, label='磁矩真实的时间演化')
plt.plot(x, tc.cat([train_pre, test_pre], dim=0), label='利用LSTM预测的磁矩随时间演化')
plt.plot([800, 800], [-0.5, 0.5], linestyle='--')
xlabel = 't'
ylabel = 'magnetic moment'
plt.legend()
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--Jx', type=float, default=0)
parser.add_argument('--Jy', type=float, default=0)
parser.add_argument('--Jz', type=float, default=1)
parser.add_argument('--hx', type=float, default=0.5)

args = parser.parse_args()
J = [args.Jx, args.Jy, args.Jz]
hx = args.hx

plt.title('t={}, V={}, tot_train_loss={:.4e}, tot_test_loss={:.4e}'.format(t, V, tc.sum(train_loss).item(), tc.sum(test_loss).item()))

plt.savefig('D:\\Manual\\Code\\tests\\Graduation Project\\pics\\l_18_J_{}_hx_{}.png'.format(J, hx), dpi=300)