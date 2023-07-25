import torch as tc
import numpy as np
from Library import PhysModule as phy
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--time', type=float, default=100)
parser.add_argument('--pt_it', type=int, default=10)
parser.add_argument('--step_by_step', '-s', type=str, default='y')
args = parser.parse_args()
interval = args.pt_it
time_tot = args.time
tau = 0.01
device = tc.device('cuda:0')

def mag_from_states(states):
    spin = phy.spin_operators('half', device='cpu')
    mag_z = list()
    for _ in range(states.shape[0]):
        mag_z.append(phy.magnetizations(states[_], [spin['sz']]))
    mag_z = tc.cat(mag_z, dim=0)
    mag_z_tot = mag_z.sum(dim=1) / mag_z.shape[1] # 不同时刻链的z方向总磁矩对链长做平均
    return mag_z_tot

# 导入数据
states_real = np.load('GraduationProject/Data/states_dt{:d}_tot{:.0f}.npy'.format(interval, time_tot), allow_pickle=True)
print(states_real[0].shape)
state0 = states_real[0]
qc = tc.load('GraduationProject/Data/qc_dt{:d}_tot{:.0f}.pth'.format(interval, time_tot))
qc.single_state = True
time_diff = 10 # 将总时长分为10份
for i in range(time_diff):
    data_real = tc.stack(list(tc.from_numpy(states_real[:int((i + 1)*time_tot/tau/interval/time_diff)])))
    print(data_real.shape)

    if args.step_by_step == 'n':
        states_pred = []
        state_temp = tc.from_numpy(state0).to(device)
        for _ in range(int(time_tot/tau/interval/time_diff*(i+1))):
            states_pred.append(state_temp)
            state_temp = qc(state_temp)
        data_pred = tc.stack(states_pred)
        file_name = 'GraduationProject/pics/1toAll_dt{:d}_tot{:.0f}.svg'.format(interval, (i+1)*time_tot/time_diff)
    elif args.step_by_step == 'y':
        states_pred = np.load('GraduationProject/Data/output_adqc_dt{:d}_tot{:.0f}.npy'.format(interval, (i+1)*time_tot/time_diff), allow_pickle=True)
        data_pred = tc.stack(list(tc.from_numpy(states_pred)))
        file_name = 'GraduationProject/pics/magz_dt{:d}_tot{:.0f}.svg'.format(interval, (i+1)*time_tot/time_diff)
    print(data_pred.shape)

    mag_z_real_tot = mag_from_states(data_real)
    mag_z_pred_tot = mag_from_states(data_pred.to('cpu'))
    x = tc.arange(0, mag_z_pred_tot.shape[0]) * 0.1

    legends = []
    plt.plot(x, mag_z_real_tot, label='real mag')
    plt.plot(x, mag_z_pred_tot.detach().numpy(), label='predicted mag')
    plt.legend()
    plt.xlabel('time/s')
    plt.ylabel('mag per site')
    plt.savefig(file_name)
    plt.close()

    # from QC2Mat import *
    # from TE2Mat import *

    # para = dict()
    # para['length'] = 10
    # para['time_it'] = interval
    # para['print_time_it'] = interval
    # para['time_tot'] = (i+1)*200
    # mat_qc = QC2Mat(para)
    # mat_te = TE2Mat(para)
    # mat_qc = mat_qc.matmat(np.eye(2**10, dtype=np.complex128))
    # mat_te = mat_te.matmat(np.eye(2**10, dtype=np.complex128))
    # loss = np.linalg.norm(mat_qc - mat_te)
    # print('print_time_it={:d}, time_tot={:.0f}, l2_mat_norm={:.6e}'.format(interval, time_tot/5*(i+1), loss))
    # print((mat_qc - mat_qc.T.conj()))
