import sys
sys.path.append(sys.path[0]+'\\Library')
print(sys.path)

import torch as tc
import numpy as np
from Library import PhysModule as phy
from matplotlib import pyplot as plt

def mag_from_states(states):
    spin = phy.spin_operators('half', device='cpu')
    mag_z = list()
    for _ in range(states.shape[0]):
        mag_z.append(phy.magnetizations(states[_], [spin['sz']]))
    mag_z = tc.cat(mag_z, dim=0)
    mag_z_tot = mag_z.sum(dim=1) / mag_z.shape[1] # 不同时刻链的z方向总磁矩对链长做平均
    return mag_z_tot


states_real = np.load('D:\\Manual\\Code\\tests\\GraduationProject\\Data\\states6.npy', allow_pickle=True)
print(states_real[0].shape)
data_real = tc.stack(list(tc.from_numpy(states_real)))
print(data_real.shape)

states_pred = np.load('D:\\Manual\\Code\\tests\\GraduationProject\\Data\\output_adqc.npy', allow_pickle=True)
print(type(states_pred))
data_pred = tc.tensor(states_pred)
print(data_pred.shape)

def fidelity(psi1, psi0):
    f = 0
    for i in range(psi1.shape[0]):
        psi0_ = psi0[i]
        psi1_ = psi1[i]
        x_pos = list(range(len(psi1_.shape)))
        y_pos = x_pos
        f += bf.tmul(psi1_.conj(), psi0_, x_pos, y_pos)
    f = f/psi1.shape[0]
    return f

mag_z_real_tot = mag_from_states(data_real)
mag_z_pred_tot = mag_from_states(data_pred)
x = tc.arange(0, mag_z_pred_tot.shape[0]) * 0.1

legends = []
plt.plot(x, mag_z_real_tot, label='real mag')
plt.plot(x, mag_z_pred_tot, label='predicted mag')
plt.legend()
plt.xlabel('time/s')
plt.ylabel('mag per site')
plt.show()