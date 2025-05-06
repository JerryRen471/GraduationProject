from copy import deepcopy
import random
import torch as tc
import numpy as np

import sys
sys.path.append('/data/home/scv7454/run/GraduationProject')

from Library import BasicFun as bf
from Library import PhysModule as phy
from Library.TensorNetwork import TensorTrain_pack as tt_pack
from Library.TEBD import *

def pure_states_evolution(states:tt_pack, gates:list, which_where:list):
    """Evolve the state by several gates0.

    :param state: initial state
    :param gates: quantum gates
    :param which_where: [which gate, which spin, which spin]
    :return: evolved state
    Example: which_where = [[0, 1, 2], [1, 0, 1]] means gate 0 on spins
    1 and 2, and gate 1 on spins 0 and 1
    """    
    for n in range(len(which_where)):
        states.act_n_body_gate(gates[which_where[n][0]], which_where[n][1:])
    return states

def PXP_mul_states_evol(states, para=None):
    para_def = dict()
    para_def['length'] = 10
    para_def['time_tot'] = 1
    para_def['tau'] = 0.01
    para_def['print_time'] = para_def['time_tot']
    para_def['device'] = tc.device('cuda:0')
    para_def['dtype'] = tc.complex64
    if para is None:
        para = dict()
    para = dict(para_def, **para)
    P = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    P[0, 0] = 1+0.j
    sigma_x = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    sigma_x[0, 1] = sigma_x[1, 0] = 1+0.j
    
    hamilt = tc.kron(P, tc.kron(sigma_x, P)).reshape([2] * 6)
    hamilt_dict = gen_1d_hamilt_dict(length=para['length'], hamilt_list=[hamilt], single_pos_list=[[0,1,2]])
    evol_states = TEBD(hamilt_dict, tau=para['tau'], time_tot=para['time_tot'], print_time=para['print_time'], init_mps=states, obs=[])
    return evol_states

def Heis_mul_states_evol(states, para=None):
    pass

def XXZ_inhomo_mul_states_evol(states, para=None):
    para_def = dict()
    para_def['J'] = 1
    para_def['delta'] = 1
    para_def['theta'] = 0
    para_def['length'] = 10
    para_def['spin'] = 'half'
    para_def['BC'] = 'open'
    para_def['time_tot'] = 1
    para_def['tau'] = 0.01
    para_def['print_time'] = para_def['time_tot']
    para_def['device'] = None
    para_def['dtype'] = tc.complex64
    if para is None:
        para = dict()
    para = dict(para_def, **para)
    para['d'] = phy.from_spin2phys_dim(para['spin'])
    para['time_it'] = round(para['time_tot'] / para['tau'])
    para['print_time'] = para['print_time'] // para['tau']
    # print(para)

    which_where = list()
    gate_list = list()
    for n in range(para['length']-1):
        Jz_n = para['delta']+para['theta']*(2*(n+1) - para['length'])/(para['length']-2)
        hamilt = phy.hamiltonian_heisenberg(para['spin'], para['J'], para['J'], Jz_n,
                                        [0, 0],
                                        [0, 0],
                                        [0, 0],
                                        device=para['device'], dtype=para['dtype'])
        gate = tc.matrix_exp(- 1j * para['tau'] * hamilt).reshape([para['d']] * 4)
        gate_list.append(gate)
        # [which gate, spin 1, spin 2]
        which_where.append([n, n, n+1])
    if para['BC'].lower() in ['pbc', 'periodic']:
        which_where.append([0, 0, para['length']-1])
    else:
        which_where[0][0] = 1
        which_where[-1][0] = 2
    list_states = list()
    states = deepcopy(states)
    for t in range(para['time_it']):
        states = pure_states_evolution(states, gate_list, which_where)
        if t % para['print_time'] == para['print_time']-1:
            list_states.append(deepcopy(states))
    return list_states

def xorX_mul_states_evol(states, para=None):
    para_def = dict()
    # model_para
    para_def['length'] = 10
    para_def['lamda'] = 1
    para_def['J'] = 1
    para_def['delta'] = 0.1
    # evol_para
    para_def['time_tot'] = 1.0
    para_def['tau'] = 0.01
    para_def['print_time'] = para_def['time_tot']
    para_def['device'] = None
    para_def['dtype'] = tc.complex64
    if para is None:
        para = dict()
    para = dict(para_def, **para)
    para['time_it'] = round(para['time_tot'] / para['tau'])
    para['print_time'] = para['print_time'] // para['tau']
    sigma_x = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    sigma_x[0, 1] = sigma_x[1, 0] = 1+0.j
    sigma_z = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    sigma_z[0, 0] = 1+0.j
    sigma_z[1, 1] = -1+0.j
    eye = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    eye[0, 0] = 1+0.j
    eye[1, 1] = 1+0.j
    hamilt_1 = para['lamda'] * (tc.kron(eye, tc.kron(sigma_x, eye)) - tc.kron(sigma_z, tc.kron(sigma_x, sigma_z))).reshape([2]*6)
    hamilt_2 = para['delta'] * sigma_z.reshape([2]*2)
    hamilt_3 = para['J'] * tc.kron(sigma_z, sigma_z).reshape([2]*4)
    hamilt = [hamilt_1, hamilt_2, hamilt_3]
    hamilt_dict = gen_1d_hamilt_dict(length=para['length'], hamilt_list=hamilt, single_pos_list=[[0,1,2], [0], [0,1]])
    evol_states = TEBD(hamilt_dict, tau=para['tau'], time_tot=para['time_tot'], print_time=para['print_time'], init_mps=states, obs=[])
    return evol_states

    # gate_1 = tc.matrix_exp(- 1j * para['tau'] * hamilt_1).reshape([2] * 6).to(dtype=para['dtype'])
    # gate_2 = tc.matrix_exp(- 1j * para['tau'] * hamilt_2).reshape([2] * 2).to(dtype=para['dtype'])
    # gate_3 = tc.matrix_exp(- 1j * para['tau'] * hamilt_3).reshape([2] * 4).to(dtype=para['dtype'])
    # gate_list = [gate_1, gate_2, gate_3]
    # which_where = []
    # for i in range(1, para['length'] - 1):
    #     # gate_list.append(gate)
    #     which_where.append([0, i-1, i, i+1])
    # for i in range(0, para['length']):
    #     which_where.append([1, i])
    # for i in range(0, para['length'] - 1):
    #     which_where.append([2, i, i+1])
    # list_states = list()
    # states = deepcopy(states)
    # for t in range(para['time_it']):
    #     states = pure_states_evolution(states, gate_list, which_where)
    #     if t % para['print_time'] == para['print_time']-1:
    #         list_states.append(deepcopy(states))
    # return list_states

def Quantum_Sun_evol(states, para=None):
    para_def = {
        'dot_size': 3,
        'length': 10,
        'alpha': 1,
        'spin': 'half',
        'time_tot': 1,
        'tau': 0.01,
        'print_time': 1,
        'device': None,
        'dtype': tc.complex64
    }
    if para is None:
        para = dict()
    para = dict(para_def, **para)
    para['d'] = phy.from_spin2phys_dim(para['spin'])
    para['time_it'] = round(para['time_tot'] / para['tau'])
    para['print_time'] = para['print_time'] // para['tau']

    gates = []

    A = tc.randn((2**para['dot_size'], 2**para['dot_size']), device=para['device'], dtype=tc.float32)
    A = A.to(para['dtype'])
    H_dot = 1/pow(2**para['dot_size'] + 1, 0.5) * 1/pow(2, 0.5) * (A + A.T)
    ops = phy.spin_operators(spin=para['spin'], device=para['device'], dtp=para['dtype'])
    SxSx = bf.kron(ops['sx'], ops['sx'])
    Sz = ops['sz']

    u = [j + 0.4*(tc.rand(1, device=para['device'])-0.5) for j in range(para['out_size'])]
    u[0] = 0
    h = [0.5 + tc.rand(1, device=para['device']) for _ in range(para['out_size'])]

    list_states = list()
    states = deepcopy(states)
    for t in range(para['time_it']):
        # states = pure_states_evolution(states, [gate, gate_l, gate_r], which_where)
        gate_dot = tc.matrix_exp(- 1j * para['tau'] * H_dot).reshape([para['d']] * 6)
        states = pure_states_evolution(states, [gate_dot], [[0, 0, 1, 2]])

        for j in range(para['out_size']):
            site_pos = para['dot_size'] + j
            dot_pos = random.randint(0, para['dot_size']-1)

            gate_interact = tc.matrix_exp(- 1j * para['tau'] * para['alpha']**u[j] * SxSx).reshape([para['d']] * 4)
            states = pure_states_evolution(states, [gate_interact], [[0, dot_pos, site_pos]])

            gate_mag = tc.matrix_exp(- 1j * para['tau'] * h[j] * Sz).reshape([para['d']] * 2)
            states = pure_states_evolution(states, [gate_mag], [[0, site_pos]])

        if t % para['print_time'] == para['print_time']-1:
            list_states.append(deepcopy(states))
    return list_states

def random_circuit(states, para=None):
    para_def = dict()
    para_def['length'] = 10
    para_def['depth'] = 4
    para_def['device'] = None
    para_def['dtype'] = tc.complex64
    if para is None:
        para = dict()
    para = dict(para_def, **para)
    H_gate = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    S_gate = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    T_gate = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    H_gate[0, 0] = H_gate[0, 1] = H_gate[1, 0] = 1/2**0.5
    H_gate[1, 1] = - 1/2**0.5
    S_gate[0, 0] = 1
    S_gate[1, 1] = 1.j
    T_gate[0, 0] = 1
    T_gate[1, 1] = 1/2**0.5 + 1.j * 1/2**0.5
    CNOT_gate = tc.zeros([4, 4], device=para['device'], dtype=para['dtype'])
    CNOT_gate[0, 0] = CNOT_gate[1, 1] = CNOT_gate[2, 3] = CNOT_gate[3, 2] = 1
    CNOT_gate = CNOT_gate.reshape([2] * 4)
    gate_list = [H_gate, S_gate, T_gate, CNOT_gate]
    which_where = []
    # @staticmethod
    # def choose_gate():
    #     r = tc.rand(1)
    #     if r < 1/3:
    #         return H_gate
    #     elif r < 2/3:
    #         return S_gate
    #     else:
    #         return T_gate
    # 用 which_where 的方式表示生成的随机线路
    for i in range(para['depth']):
        # 每一层包含一层单比特门和一层 cnot 门
        for j in range(para['length']):
            r = tc.rand(1)
            if r < 1/3:
                gate_idx = 0
            elif r < 2/3:
                gate_idx = 1
            else:
                gate_idx = 2
            which_where.append([gate_idx, j])
        for k in range((para['length'] - i % 2) // 2):
            which_where.append([3, k*2 + i%2, k*2 + i%2 + 1])
    list_states = list()
    states = deepcopy(states)
    states = pure_states_evolution(states, gate_list, which_where)
    list_states.append(states)
    return states

def main(model_name:str, model_para:dict, init_states:tt_pack, evol_para:dict, return_mat:bool=False):
    """
    Perform time evolution of quantum states.

    Parameters:
    model_name (str): Name of the model, supports 'Quantum_Sun', 'XXZ_inhomo', 'PXP', 'xorX'.
    model_para (dict): Parameters for the model.
    train_init_states (TensorTrain_pack): Initial quantum states.
    test_init_states (TensorTrain_pack): Initial quantum states.
    evol_para (dict): Parameters for evolution.

    Returns:
    list: A list containing evolved quantum states, which are in the form of TensorTrain_pack.
    
    Raises:
    ValueError: If the provided model name is invalid.
    """
    if return_mat:
        para = dict(model_para, **evol_para)
        # E = tc.eye(2**para['length'], dtype=para['dtype'], device=para['device'])
        # shape_ = [E.shape[0]] + [2] * para['length']
        # E = E.reshape(shape_)
        evol_mat_para = dict(evol_para)
        evol_mat_para['time_tot'] = evol_para['print_time']

        if model_name == 'Quantum_Sun':
            evol_states = Quantum_Sun_evol(init_states, dict(model_para, **evol_para))
            # evol_mat = Quantum_Sun_evol(E, dict(model_para, **evol_mat_para)).reshape(E.shape[0], -1)
        elif model_name == 'XXZ_inhomo':
            evol_states = XXZ_inhomo_mul_states_evol(init_states, dict(model_para, **evol_para))
            # evol_mat = XXZ_inhomo_mul_states_evol(E, dict(model_para, **evol_mat_para)).reshape(E.shape[0], -1)
        elif model_name == 'PXP':
            evol_states = PXP_mul_states_evol(init_states, dict(model_para, **evol_para))
            # evol_mat = PXP_mul_states_evol(E, dict(model_para, **evol_mat_para)).reshape(E.shape[0], -1)
        elif model_name == 'xorX':
            evol_states = xorX_mul_states_evol(init_states, dict(model_para, **evol_para))
            # evol_mat = xorX_mul_states_evol(E, dict(model_para, **evol_mat_para)).reshape(E.shape[0], -1)
        elif model_name == 'random_circuit':
            evol_states = random_circuit(init_states, dict(model_para, **evol_para))
            # evol_mat = random_circuit(E, dict(model_para, **evol_mat_para)).reshape(E.shape[0], -1)
        else:
            raise ValueError
        return evol_states
    
    else:
        if model_name == 'Quantum_Sun':
            evol_states = Quantum_Sun_evol(init_states, dict(model_para, **evol_para))
        elif model_name == 'XXZ_inhomo':
            evol_states = XXZ_inhomo_mul_states_evol(init_states, dict(model_para, **evol_para))
        elif model_name == 'PXP':
            evol_states = PXP_mul_states_evol(init_states, dict(model_para, **evol_para))
        elif model_name == 'xorX':
            evol_states = xorX_mul_states_evol(init_states, dict(model_para, **evol_para))
        elif model_name == 'random_circuit':
            evol_states = random_circuit(init_states, dict(model_para, **evol_para))
        else:
            raise ValueError
        return evol_states
    # train_label = evol_states
    # train_input = tc.cat((train_init_states.unsqueeze(1), train_evol_states[:, :-1]), dim=1)
    # train_input = tc.cat((test_init_states.unsqueeze(1), test_evol_states[:, :-1]), dim=1)
    # test_label = test_evol_states
    # test_input = tc.cat((test_init_states.unsqueeze(1), test_evol_states[:, :-1]), dim=1)
    # test_input = tc.cat((test_init_states.unsqueeze(1), test_evol_states[:, :-1]), dim=1)
    # merge = lambda x: x.reshape(x.shape[0]*x.shape[1], *x.shape[2:])
    # data = dict()
    # data['train_set'] = merge(train_input)
    # data['train_label'] = merge(train_label)
    # data['test_set'] = merge(test_input)
    # data['test_label'] = merge(test_label)
    # return data