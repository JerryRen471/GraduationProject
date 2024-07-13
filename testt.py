import torch as tc
from Library import PhysModule as phy
from Library.TensorNetwork import TensorTrain
import time

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
    t1 = time.time()
    device = tc.device('cpu')
    dtype = tc.complex64
    length = 19
    t = tc.rand([2**length], dtype=dtype)
    # t[0] = 1
    t = t.reshape([2]*length)
    mps = TensorTrain(t, device=device, dtype=dtype, center=2, chi=3)
    para = dict()
    para = {'spin':'half', 'J':[1, 1, 1], 'h':[0, 0, 1], 'hl':0, 'device':device, 'dtype':dtype, 'tau':0.01, 'd':2}
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
    for t in range(100):
        evol_gates(length, mps, gate, gate_l=gate_l, gate_r=gate_r)
    # states = tc.stack(list_states, dim=1)
    # evol_gates(length, mps, gate, gate_l, gate_r)
    t2 = time.time()
    print(t2-t1)