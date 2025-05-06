import torch as tc
from Library.TEBD import *
import Library.TensorNetwork as TN
from TN_WorkFlow.TimeEvol import PXP_mul_states_evol as PXP_mps
from WorkFlow.TimeEvol import PXP_mul_states_evl as PXP_states
from WorkFlow.InitStates import rand_states, rand_dir_prod_states

length = 5
number = 7
dtype = tc.complex64
device = tc.device('cuda:0')
# init_states = rand_dir_prod_states(length=length, number=number, dtype=dtype, device=device)
# init_mps = TN.TensorTrain_pack(tensor_packs=[init_states], length=length, phydim=2, chi=None, device=device, dtype=dtype, initialize=True)

para = {
    'length': length,
    'tau': 0.02,
    'time_tot': 0.2,
    'print_time': 0.02,
    'device': tc.device('cuda:0')
}
print(para['print_time'])
print(para['time_tot'])

import time
length_list = [i for i in range(6, 21)]
t_normal = []
t_mps = []
number = 10
device = tc.device('cuda:0')
for length in length_list:
    print('length=', length)
    para['length'] = length
    init_states = rand_dir_prod_states(length=length, number=number, dtype=dtype, device=device)
    init_mps = TN.TensorTrain_pack(tensor_packs=[init_states], length=length, phydim=2, chi=None, device=device, dtype=dtype, initialize=True)
    t1 = time.time()
    evol_states = PXP_states(states=init_states, para=para)
    t2 = time.time()
    evol_mps = PXP_mps(states=init_mps, para=para)
    t3 = time.time()
    print('-'*20)
    print('without TEBD', t2-t1)
    print('TEBD', t3-t2)
    t_normal.append(t2-t1)
    t_mps.append(t3-t2)
print('#'*20)
print('#'*20)
print('#'*20)

print(t_normal)
print(t_mps)