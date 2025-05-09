import Library.TensorNetwork as TN
import torch as tc
import numpy as np

def multiply_list(list):
    res = 1
    for i in list:
        res *= i
    return res

def TEBD(gate_dict:dict, tau:float, time_tot:float, print_time:float, init_mps:TN.TensorTrain_pack, obs:list):
    '''
    Hamiltonian contains {'gate_i':[gate_0, gate_1, gate_2, ...], 'pos':[pos0, pos1, pos2, ...]}
    tau is the time step
    t_steps is a list of time steps that are to be printed
    init_mps is the initial state
    obs is a list of observables, each observable is a list of operators and their positions
        Format: [[op1, pos1], [op2, pos2], ...]
    '''
    gate_list = gate_dict['gate_i']
    pos = gate_dict['pos']
    total_t_list = [i for i in range(1, int(time_tot // tau) + 1)]
    print_t = int(print_time // tau)
    mps_evol = []
    obs_evol = []

    for t in total_t_list:
        for i, gate_i in enumerate(gate_list):
            init_mps = init_mps.act_n_body_gate_sequence(gate_i, pos[i], set_center=0)
        if t % print_t == 0:
            mps_evol.append(TN.copy_from_mps_pack(init_mps))
            # Calculate expected values of observables
            obs_values = []
            for op, op_pos in obs:
                if isinstance(op_pos, int):
                    # Single-site observable
                    obs_value = TN.inner_mps_pack(init_mps, init_mps.act_single_site_op(op, op_pos))
                else:
                    # Multi-site observable
                    obs_value = TN.inner_mps_pack(init_mps, init_mps.act_n_body_gate(op, op_pos))
                obs_values.append(obs_value)
            obs_evol.append(obs_values)
    return mps_evol, obs_evol

def gen_1d_pos_sequence(length:int, single_pos:list, repeat_interval:int=1):
    tmp_pos = single_pos[:]
    pos_sequence = []
    while length > tmp_pos[-1]:
        pos_sequence.append(tmp_pos)
        tmp_pos = [i + repeat_interval for i in tmp_pos]
    rev_seq = list(pos_sequence.__reversed__())
    pos_sequence = pos_sequence[:] + rev_seq[:]
    return pos_sequence

def gen_1d_gate_dict(length, hamilt_list, tau:float, single_pos_list, repeat_interval:int=1, bc='open'):
    return_pos = []
    gate_list = []
    for i, hamilt in enumerate(hamilt_list):
        shape = list(hamilt.shape)
        mid = len(shape)//2
        mat_shape = [multiply_list(shape[:mid]), multiply_list(shape[mid:])]
        gate_i = tc.matrix_exp(-1j * hamilt.reshape(mat_shape) * tau / 2).reshape(shape)
        gate_list.append(gate_i)
        return_pos.append(gen_1d_pos_sequence(length, single_pos_list[i], repeat_interval))
    return {'gate_i':gate_list, 'pos':return_pos}