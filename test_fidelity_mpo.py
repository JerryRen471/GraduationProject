import torch as tc
import numpy as np
from Library.TensorNetwork import n_body_gate_to_mpo

def fill_seq(pos:list):
    delta_pos = list(i for i in range(pos[0], pos[-1]+1))
    for i in pos:
        delta_pos.remove(i)
    return delta_pos
def step_function(i:int, pos:list):
    f = lambda x: 0 if x < 0 else 1
    y = 0
    for j in pos[1:]:
        y = y + f(i - j)
    return y

def cal_gate_fidelity_from_mpo(gates, which_where, n_qubit, device=tc.device('cpu'), dtype=tc.complex64):
    mpo = list(None for _ in range(n_qubit))
    occupied = list(None for _ in range(n_qubit))
    for i_pos in which_where:
        gate = gates[i_pos[0]]
        pos = i_pos[1:]
        mpo_t_list = n_body_gate_to_mpo(gate=gate, n=len(pos), device=device, dtype=dtype)
        # print(len(gate_list))
        delta_dim_list = [mpo_t_list[step_function(i, pos)].shape[-1] for i in range(pos[0], pos[-1]+1)]
        delta_pos = fill_seq(pos)

        gate_idx = 0
        for i in range(pos[0]+1, pos[-1]):
            mpo_i = mpo[i]
            
            if i in pos:
                if mpo_i == None:
                    mpo[i] = mpo_t_list[gate_idx]
                else:
                    mpo[i] = tc.einsum()
            elif i in delta_pos:
                delta_dim = delta_dim_list[i]
                delta = tc.einsum('il, jl -> ijkl', tc.eys(delta_dim, device=device, dtype=dtype), tc.eys(2, device=device, dtype=dtype))



