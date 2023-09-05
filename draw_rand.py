import torch as tc
import numpy as np
from Library import PhysModule as phy
from matplotlib import pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--folder', type=str, default='rand_init/')
parser.add_argument('--train_num', type=int, default=100)
args = parser.parse_args()
train_num = args.train_num

results = np.load('GraduationProject/Data/'+args.folder+'adqc_result_num{:d}.npy'.format(args.train_num), allow_pickle=True) # results是字典, 包含'train_pred', 'test_pred', 'train_loss', 'test_loss'
results = results.item()
train_loss = results['train_loss']
test_loss = results['test_loss']
x = list(range(0, len(train_loss)))

print('train_num={:d}\ntrain_loss:{:.4e}\ttest_loss:{:.4e}\n'.format(train_num, train_loss[-1], test_loss[-1]))

legends = []
plt.plot(x, train_loss, label='train loss')
plt.plot(x, test_loss, label='test loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('GraduationProject/pics/'+args.folder+'loss_num{:d}.svg'.format(args.train_num))
plt.close()