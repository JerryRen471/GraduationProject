import torch as tc
import numpy as np

from rand_PXP import *

def entanglement_entropy(states):
    length = len(states.shape) - 1
    num = states.shape[0]
    shape = [num, int(2**(length//2)), int(2**(length - length//2))]
    states_ = states.reshape(shape)
    spectral = tc.linalg.svdvals(states_)
    p_distribution = spectral**2
    # print(p_distribution.shape)
    entropy = tc.einsum("ij,ij->i", -p_distribution, tc.log2(p_distribution))
    return entropy

# 生成一组直积态，用PXP模型进行演化，计算演化过程中的二分纠缠熵的变化

init_states = rand_dir_prod_states(10, 10, device=tc.device('cpu'), dtype=tc.complex64)

para = {
    'length': 10,
    'tau': 0.02,
    'time_tot': 100,
    'print_time': 0.02,
    'device': tc.device('cpu'),
    'dtype': tc.complex64
}

evol_states = PXP_mul_states_evl(init_states, para)
entropy_list = []

sample_num = 10
for i in range(sample_num):
    entropy = entanglement_entropy(evol_states[i])
    entropy_list.append(entropy)

entropy_stack = tc.stack(entropy_list, dim=0)
print(entropy_stack.shape)

from matplotlib import pyplot as plt
plt.plot([0.02 * i for i in range(1, entropy_stack.shape[1]+1)], entropy_stack.T)
plt.xscale('log')
plt.yscale('log')
plt.show()
plt.savefig('entropy_evol.svg')