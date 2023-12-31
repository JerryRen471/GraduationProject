import torch as tc
import numpy as np
from Library import PhysModule as phy
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--time', type=float, default=100)
parser.add_argument('--pt_it', type=int, default=10)
args = parser.parse_args()
interval = args.pt_it
time_tot = args.time

def mag_from_states(states):
    spin = phy.spin_operators('half', device='cpu')
    mag_z = list()
    for _ in range(states.shape[0]):
        mag_z.append(phy.magnetizations(states[_], [spin['sz']]))
    mag_z = tc.cat(mag_z, dim=0)
    mag_z_tot = mag_z.sum(dim=1) / mag_z.shape[1] # 不同时刻链的z方向总磁矩对链长做平均
    return mag_z_tot

# 对磁场预测作图
states_pred = np.load('GraduationProject/Data/output_adqc_dt{:d}_tot{:.0f}.npy'.format(interval, time_tot), allow_pickle=True)
print(states_pred[0].shape)
data_pred = tc.stack(list(tc.from_numpy(states_pred)))
print(data_pred.shape)

states_real = np.load('GraduationProject/Data/states_dt{:d}_tot{:.0f}.npy'.format(interval, time_tot), allow_pickle=True)
print(states_real[0].shape)
data_real = tc.stack(list(tc.from_numpy(states_real)))
print(data_real.shape)

mag_z_real_tot = mag_from_states(data_real)
mag_z_pred_tot = mag_from_states(data_pred)
x = tc.arange(0, mag_z_pred_tot.shape[0]) * 0.1

legends = []
plt.plot(x, mag_z_real_tot, label='real mag')
plt.plot(x, mag_z_pred_tot, label='predicted mag')
plt.legend()
plt.xlabel('time/s')
plt.ylabel('mag per site')
plt.savefig('GraduationProject/pics/magz_dt{:d}_tot{:.0f}.svg'.format(interval, time_tot))
plt.close()

# 对loss（非保真度）作图
train_loss = np.load('GraduationProject/Data/train_loss_dt{:d}_tot{:.0f}.npy'.format(interval, time_tot), allow_pickle=True)
test_loss = np.load('GraduationProject/Data/test_loss_dt{:d}_tot{:.0f}.npy'.format(interval, time_tot), allow_pickle=True)
x = list(range(len(train_loss)))

legends = []
plt.plot(x, train_loss, label='train loss')
plt.plot(x, test_loss, label='test loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('GraduationProject/pics/loss_dt{:d}_tot{:.0f}.svg'.format(interval, time_tot))
plt.close()