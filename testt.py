import torch as tc
from Library import PhysModule as phy
from Library.TensorNetwork import *
from rand_chaos import XXZ_inhomo_mul_states_evl
import time

from TN_rand_time_sequence import Heisenberg_mul_states_evl, label_generate

def evol_gates(length, mps, gate, gate_l, gate_r):
    for i in range(length-1):
        if i == 0:
            mps.act_two_body_gate(gate=gate_l, pos=[i, i+1])
        elif i == length-2:
            mps.act_two_body_gate(gate=gate_r, pos=[i, i+1])
        else:
            mps.act_two_body_gate(gate=gate, pos=[i, i+1])
        # print(mps.get_norm())

def rand_prod_mps_pack(number, length, chi, phydim=2, device=tc.device('cpu'), dtype=tc.complex64):
    states = [tc.rand([number, phydim], dtype=dtype) for _ in range(length)]
    for i in range(len(states)):
        site = states[i]
        norm = tc.sqrt(tc.einsum('ni, ni->n', site, site.conj()))
        site[:, 0] = site[:, 0] / norm
        site[:, 1] = site[:, 1] / norm
        states[i] = site
    # state[0] += 1
    # state = state / tc.norm(state)
    # t[0] = 1
    # state = state.reshape([2]*length)
    mps = TensorTrain_pack(states, length=length, phydim=phydim, center=-1, chi=chi, device=device, dtype=dtype)
    return mps

if __name__ == '__main__':
    device = tc.device('cpu')
    dtype = tc.complex64
    length = 6
    mps = rand_prod_mps_pack(2, length=length, chi=10, device=device, dtype=dtype)
    print("trunc_error", mps.trunc_error)
    para = dict()
    # para = {'spin':'half', 'J':[1, 1, 1], 'h':[0, 0, 0], 'hl':0, 'device':device, 'dtype':dtype, 'tau':0.01, 'd':2}
    # list_states = list()
    t1 = time.time()
    para = {'J':[1, 1, 1], 'h':[0,0,0], 'hl':0, 'length':length, 'time_tot':0.1, 'tau':0.01, 'print_time':0.01, 'device':device, 'dtype':dtype}
    input, label = label_generate(mps, para)
    print(input.node_list[0].shape)
    print(label.node_list[0].shape)