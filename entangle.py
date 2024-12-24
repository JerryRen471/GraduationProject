import torch as tc
import numpy as np

from model_evol import PXP_mul_states_evl
from gen_init import *

def entanglement_entropy(states):
    length = len(states.shape) - 1
    num = states.shape[0]
    shape = [num, int(2**(length/2)), int(2**(length - length/2))]
    states_ = states.reshape(shape)
    spectral = tc.linalg.svdvals(states_)
    p_distribution = spectral**2
    # print(p_distribution.shape)
    entropy = tc.einsum("ij,ij->i", -p_distribution, tc.log2(p_distribution))
    return entropy

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--length', type=int, default=10)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--evol_num', type=int, default=100)
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
para['length'] = args.length
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

init_states = rand_dir_prod_states(para_train['sample_num'], para['length'], device=para['device'], dtype=para['dtype'], entangle_dim=args.entangle_dim)
evol_states = PXP_mul_states_evl(init_states, para_train).squeeze(0)
print(evol_states.shape)

import matplotlib.pyplot as plt

init_ee = entanglement_entropy(init_states)
evol_ee = entanglement_entropy(evol_states)
print(init_ee)
print(evol_ee)

plt.plot(evol_ee)
plt.xlabel('Evolution Time(0.02)')
plt.ylabel('Entanglement Entropy')
plt.title('Evolution of Entanglement Entropy')
plt.savefig('/data/home/scv7454/run/GraduationProject/pics/PXP/Evolution of Entanglement Entropy.svg')
plt.close()

entangle_ee_list = []
for entangle_dim in range(33):
    entangle_states = rand_entangled_states(1, para['length'], entangle_dim, device=para['device'], dtype=para['dtype'])
    entangle_ee = entanglement_entropy(entangle_states)
    entangle_ee_list.append(entangle_ee)
    print('entangle_dim={}, ee={}'.format(entangle_dim, entangle_ee))

plt.plot(entangle_ee_list)
plt.xlabel('Entanglement Dimension')
plt.ylabel('Entanglement Entropy')
plt.title('Entanglement Entropy vs Entanglement Dimension')
plt.savefig('/data/home/scv7454/run/GraduationProject/pics/PXP/Entanglement Entropy of Different Entangle Dimension.svg')
plt.grid(True)
plt.close()

x = np.linspace(0.001, 0.999, 100)
y = -x*np.log2(x) - (1-x)*np.log2(1-x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph of y = -x*log2(x) - (1-x)*log2(1-x)')
plt.savefig('/data/home/scv7454/run/GraduationProject/pics/PXP/FunctionGraph.svg')
plt.close()
