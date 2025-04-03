import scipy as sp
import torch as tc
import numpy as np

from WorkFlow import InitStates, TimeEvol

def entanglement_entropy(states):
    length = len(states.shape) - 1
    num = states.shape[0]
    shape = [num, int(2**(length/2)), int(2**(length - length/2))]
    states_ = states.reshape(shape)
    spectral = tc.linalg.svdvals(states_)
    print(spectral.shape)
    entropy_list = []
    for i in range(spectral.shape[0]):
        p_dis = spectral[i][spectral[i] != 0]**2
        entropy = tc.einsum("i,i->", -p_dis, tc.log2(p_dis))
        entropy_list.append(entropy)
    # spectral = spectral[spectral != 0]
    # p_distribution = spectral**2
    # # print(p_distribution.shape)
    # entropy = tc.einsum("ij,ij->i", -p_distribution, tc.log2(p_distribution))
    return entropy_list

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--length', type=int, default=10)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--evol_num', type=int, default=50)
parser.add_argument('--test_num', type=int, default=500)
parser.add_argument('--entangle_dim', type=int, default=1)
parser.add_argument('--folder', type=str, default='mixed_rand_states/')
parser.add_argument('--evol_mat_path', type=str, default="GraduationProject/Data/evol_mat.npy")
parser.add_argument('--gen_type', type=str, default='d')
parser.add_argument('--time_interval', type=float, default=0.02)
parser.add_argument('--tau', type=float, default=0.02)
args = parser.parse_args()

device = tc.device('cpu')
dtype = tc.complex128

para = dict()
para['length'] = 6
para['device'] = device
para['gen_type'] = args.gen_type
para['spin'] = 'half'
para['d'] = 2
para['dtype'] = dtype
para['tau'] = args.tau

para_train = dict(para)
para_train['time_tot'] = args.time_interval*args.evol_num
para_train['print_time'] = args.time_interval
para_train['sample_num'] = args.sample_num

import matplotlib.pyplot as plt
from Library.PhysModule import multi_mags_from_states

fig, ax1 = plt.subplots()
fig, ax2 = plt.subplots()

ksi_list = [0.1 * _ for _ in range(11)]
evol_ee_list = []
for ksi in ksi_list:
    init_states = InitStates.RK_initial_state(ksi=ksi, length=para['length'], device=para['device'], dtype=para['dtype'])
    evol_states = TimeEvol.xorX_mul_states_evl(init_states, para_train).squeeze(0)
    print(evol_states.shape)

    print(init_states)
    init_ee = entanglement_entropy(init_states)
    evol_ee = entanglement_entropy(evol_states)
    mags = multi_mags_from_states(states=evol_states, device=para['device'])
    print(mags.shape)
    ax2.plot(mags[:, 0, 4], label='ksi={}'.format(ksi))
    evol_ee = tc.tensor(evol_ee, device=para['device'])
    print(init_ee)
    print(evol_ee)
    ax1.plot(evol_ee, label='ksi={}'.format(ksi))
    evol_ee_list.append(evol_ee)

evol_ee_plt = tc.stack(evol_ee_list, dim=0)
print(evol_ee_plt.shape)

# init_states = InitStates.rand_dir_prod_states(number=1, length=para['length'], device=para['device'], dtype=para['dtype'])
# evol_states = TimeEvol.xorX_mul_states_evl(init_states, para_train).squeeze(0)
# evol_ee = entanglement_entropy(evol_states)
# ax1.plot(evol_ee)

# init_states = InitStates.eig_states(delta=0.1, lamda=1, J=1, number=1, length=para['length'], device=para['device'], dtype=para['dtype'])
# evol_states = TimeEvol.xorX_mul_states_evl(init_states, para_train).squeeze(0)
# evol_ee = entanglement_entropy(evol_states)
# ax1.plot(evol_ee)

ax1.set_xlabel('Evolution Time(0.02)')
ax1.set_ylabel('Entanglement Entropy')
ax1.set_title('Evolution of Entanglement Entropy')
ax1.legend()
fig1 = ax1.figure
fig1.savefig('/data/home/scv7454/run/GraduationProject/pics/xorX/Evolution of Entanglement Entropy(RK_init).svg')

ax2.set_xlabel('Evolution Time(0.02)')
ax2.set_ylabel('Mag_x on Site 5')
ax2.set_title('Evolution of Mag_x on Site 5')
ax2.legend()
fig2 = ax2.figure
fig2.savefig('/data/home/scv7454/run/GraduationProject/pics/xorX/Evolution of Mag_x on Site 5(RK_init).svg')


plt.close()

# entangle_ee_list = []
# for entangle_dim in range(33):
#     entangle_states = rand_entangled_states(1, para['length'], entangle_dim, device=para['device'], dtype=para['dtype'])
#     entangle_ee = entanglement_entropy(entangle_states)
#     entangle_ee_list.append(entangle_ee)
#     print('entangle_dim={}, ee={}'.format(entangle_dim, entangle_ee))

# plt.plot(entangle_ee_list)
# plt.xlabel('Entanglement Dimension')
# plt.ylabel('Entanglement Entropy')
# plt.title('Entanglement Entropy vs Entanglement Dimension')
# plt.savefig('/data/home/scv7454/run/GraduationProject/pics/PXP/Entanglement Entropy of Different Entangle Dimension.svg')
# plt.grid(True)
# plt.close()

# x = np.linspace(0.001, 0.999, 100)
# y = -x*np.log2(x) - (1-x)*np.log2(1-x)

# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Graph of y = -x*log2(x) - (1-x)*log2(1-x)')
# plt.savefig('/data/home/scv7454/run/GraduationProject/pics/PXP/FunctionGraph.svg')
# plt.close()
