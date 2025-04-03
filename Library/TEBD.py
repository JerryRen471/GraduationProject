import Library.TensorNetwork as TN
import torch as tc
import numpy as np

def multiply_list(list):
    res = 1
    for i in list:
        res *= i
    return res

def TEBD(Hamiltonian:dict, tau:float, t_steps:list, init_mps:TN.TensorTrain_pack, obs:list):
    '''
    Hamiltonian contains {'Hi':[H0, H1, H2, ...], 'pos':[pos0, pos1, pos2, ...]}
    tau is the time step
    t_steps is a list of time steps that are to be printed
    init_mps is the initial state
    obs is a list of observables
    '''
    Hi = Hamiltonian['Hi']
    pos = Hamiltonian['pos']
    total_t_list = [i for i in range(int(t_steps[-1] // tau) + 1)]
    mps_evol = []
    for t in total_t_list:
        for i, hi in enumerate(Hi):
            shape = list(hi.shape)
            mid = len(shape)//2
            mat_shape = [multiply_list(shape[:mid]), multiply_list(shape[mid:])]
            gate_i = tc.matrix_exp(-1j * hi.reshape(mat_shape) * tau / 2).reshape(shape)
            init_mps = init_mps.act_n_body_gate_sequence(gate_i, pos[i], set_center=0)
            print('-'*20)
            print('t = ', t * tau)
            for node_idx, node in enumerate(init_mps.node_list):
                print('node_', node_idx, '.shape=', node.shape)
        # if t in t_steps:
        #     print(t * tau)
        #     mps_evol.append(TN.copy_from_mps_pack(init_mps))
    return mps_evol

def gen_1d_pos_sequence(length:int, single_pos:list, repeat_interval:int=1):
    tmp_pos = single_pos[:]
    pos_sequence = []
    while length > tmp_pos[-1]:
        pos_sequence.append(tmp_pos)
        tmp_pos = [i + repeat_interval for i in tmp_pos]
    rev_seq = list(pos_sequence.__reversed__())
    pos_sequence = pos_sequence[:] + rev_seq[:]
    return pos_sequence

def gen_1d_hamilt_dict(length, hamilt_list, single_pos_list, repeat_interval:int=1, bc='open'):
    return_pos = []
    for i, hamilt in enumerate(hamilt_list):
        return_pos.append(gen_1d_pos_sequence(length, single_pos_list[i], repeat_interval))
    return {'Hi':hamilt_list, 'pos':return_pos}