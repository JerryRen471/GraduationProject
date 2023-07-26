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
tau = 0.01
device = tc.device('cuda:0')
dtype = tc.complex128
print('=='*10)
print('print interval = {}, total time = {}'.format(interval, time_tot))

qc = tc.load('GraduationProject/Data/qc_dt{:d}_tot{:.0f}.pth'.format(interval, time_tot))
qc.single_state = True
states_real = np.load('GraduationProject/Data/states_dt{:d}_tot{:.0f}.npy'.format(interval, time_tot), allow_pickle=True)
state0 = states_real[0]
states_pred = []
state_temp = tc.from_numpy(state0).to(device)
for _ in range(int(time_tot / tau / interval)):
    states_pred.append(state_temp)
    state_temp = qc(state_temp)

data_pred = tc.stack(states_pred)
data_real = tc.stack(list(tc.from_numpy(states_real)))
print(data_pred.shape)
print(data_real.shape)

def mag_from_states(states):
    spin = phy.spin_operators('half', device='cpu')
    mag_z = list()
    for _ in range(states.shape[0]):
        mag_z.append(phy.magnetizations(states[_], [spin['sz']]))
    mag_z = tc.cat(mag_z, dim=0)
    mag_z_tot = mag_z.sum(dim=1) / mag_z.shape[1] # 不同时刻链的z方向总磁矩对链长做平均
    return mag_z_tot

mag_z_real_tot = mag_from_states(data_real)
mag_z_pred_tot = mag_from_states(data_pred.to('cpu'))
x = tc.arange(0, mag_z_pred_tot.shape[0]) * 0.1

legends = []
plt.plot(x, mag_z_real_tot, label='real mag')
plt.plot(x, mag_z_pred_tot.detach().numpy(), label='predicted mag')
plt.legend()
plt.xlabel('time/s')
plt.ylabel('mag per site')
plt.savefig('GraduationProject/pics/1toAll_dt{:d}_tot{:.0f}.svg'.format(interval, time_tot))
plt.show()

from QC2Mat import *
from TE2Mat import *

para = dict()
para['length'] = 10
para['time_it'] = interval
para['print_time_it'] = interval
para['time_tot'] = time_tot
mat_qc = QC2Mat(para)
mat_te = TE2Mat(para)
mat_qc = mat_qc.matmat(np.eye(2**10, dtype=np.complex128))
mat_te = mat_te.matmat(np.eye(2**10, dtype=np.complex128))
loss = np.linalg.norm(mat_qc - mat_te)
print('print_time_it={:d}, time_tot={:.0f}, l2_mat_norm={:.6e}'.format(interval, time_tot, loss))
# print((mat_qc - mat_qc.T.conj()))
