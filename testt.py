import torch as tc
from Library import PhysModule as phy
from Library.TensorNetwork import *
from rand_chaos import XXZ_inhomo_mul_states_evl
import time

from rand_time_sequence import Heisenberg_mul_states_evl

def evol_gates(length, mps, gate, gate_l, gate_r):
    for i in range(length-1):
        if i == 0:
            mps.act_two_body_gate(gate=gate_l, pos=[i, i+1])
        elif i == length-2:
            mps.act_two_body_gate(gate=gate_r, pos=[i, i+1])
        else:
            mps.act_two_body_gate(gate=gate, pos=[i, i+1])
        # print(mps.get_norm())

if __name__ == '__main__':
    device = tc.device('cpu')
    dtype = tc.complex64
    length = 6
    states = [tc.rand([5, 2], dtype=dtype) for _ in range(length)]
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
    mps = TensorTrain_pack(states, length=length, phydim=2, center=0, chi=15, device=device, dtype=dtype)
    print("trunc_error", mps.trunc_error)
    # mps1_state = copy_from_tn(mps)
    # # mps1_state.merge_all()
    # mps1_state = mps1_state.node_list[0]
    # print('mps1_state.shape', mps1_state.shape)
    # print('state.shape', state.shape)
    # print('mps1_state norm: ', tc.norm(mps1_state))
    # print('state norm: ', tc.norm(state))
    # print(tc.dist(mps1_state.squeeze(), state))
    para = dict()
    para = {'spin':'half', 'J':[1, 1, 1], 'h':[0, 0, 0], 'hl':0, 'device':device, 'dtype':dtype, 'tau':0.01, 'd':2}
    hamilt = phy.hamiltonian_heisenberg(para['spin'], para['J'][0], para['J'][1], para['J'][2],
                                        [para['h'][0]/2, para['h'][0]/2],
                                        [para['h'][1]/2, para['h'][1]/2],
                                        [para['h'][2]/2, para['h'][2]/2],
                                        device=para['device'], dtype=para['dtype'])
    gate = tc.matrix_exp(- 1j * para['tau'] * hamilt)
    # print(hamilt)
    # print(gate)

    hamilt_l = phy.hamiltonian_heisenberg(para['spin'], para['J'][0], para['J'][1], para['J'][2],
                                          [para['h'][0]+para['hl'], para['h'][0] / 2],
                                          [para['h'][1], para['h'][1] / 2],
                                          [para['h'][2], para['h'][2] / 2],
                                          device=para['device'], dtype=para['dtype'])
    gate_l = tc.matrix_exp(- 1j * para['tau'] * hamilt_l)
    # print(hamilt_l)
    # print(gate_l)

    hamilt_r = phy.hamiltonian_heisenberg(para['spin'], para['J'][0], para['J'][1], para['J'][2],
                                          [para['h'][0] / 2, para['h'][0]],
                                          [para['h'][1] / 2, para['h'][1]],
                                          [para['h'][2] / 2, para['h'][2]],
                                          device=para['device'], dtype=para['dtype'])
    gate_r = tc.matrix_exp(- 1j * para['tau'] * hamilt_r)

    # list_states = list()
    t1 = time.time()
    for t in range(10):
        evol_gates(length, mps, gate, gate_l=gate_l, gate_r=gate_r)
        print(f"{t}evol: trunc_error", mps.trunc_error)
    # mps2 = TensorNetwork(chi=3, device=device, dtype=dtype)
    t2 = time.time()
    tc.save(mps, 'test.pt')
    mps2 = mps
    mps2.merge_all()
    mps2state = mps2.node_list[0]
    # states = tc.stack(list_states, dim=1)
    # evol_gates(length, mps, gate, gate_l, gate_r)
    print(t2-t1)
    prod_states = states[0]
    prod_states = tc.einsum('ni,nk->nik', prod_states, states[1])
    for i in states[2:]:
        prod_states = tc.einsum('ni...j,nk->ni...jk', prod_states, i)
    print(prod_states.shape)
    para = {'J':[1, 1, 1], 'h':[0,0,0], 'hl':0, 'length':length, 'time_tot':0.1, 'tau':0.01, 'print_time':0.1, 'device':device, 'dtype':dtype}
    t3 = time.time()
    states2 = Heisenberg_mul_states_evl(prod_states, para)
    t4 = time.time()
    # state2 = states2[0]
    print(states2.squeeze().shape)
    print(mps2state.shape)
    print(tc.dist(states2.squeeze(), mps2state.squeeze()))
    print('mps time', t2-t1)
    print('state time', t4-t3)