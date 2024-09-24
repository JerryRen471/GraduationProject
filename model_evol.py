import torch as tc
import numpy as np
from Library import BasicFun as bf
from Library import PhysModule as phy
import random

def pure_states_evolution(states:tc.Tensor, gates:list, which_where:list)->tc.Tensor:
    """Evolve the state by several gates0.

    :param state: initial state
    :param gates: quantum gates
    :param which_where: [which gate, which spin, which spin]
    :return: evolved state
    Example: which_where = [[0, 1, 2], [1, 0, 1]] means gate 0 on spins
    1 and 2, and gate 1 on spins 0 and 1
    """
    def pure_states_evolution_one_gate(v, g, pos):
        ind = list(range(len(pos), 2*len(pos)))
        pos = [_+1 for _ in pos]
        # print("v.dtype",v.dtype)
        # print("g.dtype",g.dtype)
        v = tc.tensordot(v, g, [pos, ind]) # 对每个pos加一
        ind0 = list(range(v.ndimension()))
        for nn in range(len(pos)):
            ind0.remove(pos[nn])
        ind0 += pos
        order = np.argsort(ind0)
        return v.permute(tuple(order))

    for n in range(len(which_where)):
        states = pure_states_evolution_one_gate(
            states, gates[which_where[n][0]], which_where[n][1:])
    return states

def PXP_mul_states_evl(states, para=None):
    para_def = dict()
    para_def['length'] = 10
    para_def['time_tot'] = 1
    para_def['tau'] = 0.01
    para_def['print_time'] = para_def['time_tot']
    para_def['device'] = None
    para_def['dtype'] = tc.complex64
    if para is None:
        para = dict()
    para = dict(para_def, **para)
    para['time_it'] = round(para['time_tot'] / para['tau'])
    para['print_time'] = para['print_time'] // para['tau']
    para['device'] = bf.choose_device(para['device'])
    P = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    P[0, 0] = 1+0.j
    sigma_x = tc.zeros([2, 2], device=para['device'], dtype=para['dtype'])
    sigma_x[0, 1] = sigma_x[1, 0] = 1+0.j
    hamilt = tc.kron(P, tc.kron(sigma_x, P))
    gate = tc.matrix_exp(- 1j * para['tau'] * hamilt).reshape([2] * 6).to(dtype=para['dtype'])
    gate_list = [gate]
    which_where = []
    for i in range(1, para['length'] - 1):
        # gate_list.append(gate)
        which_where.append([0, i-1, i, i+1])
    list_states = list()
    for t in range(para['time_it']):
        states = pure_states_evolution(states, gate_list, which_where)
        if t % para['print_time'] == para['print_time']-1:
            list_states.append(states)
    states = tc.stack(list_states, dim=1)
    return states

def XXZ_inhomo_mul_states_evl(states, para=None):
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
    para['device'] = bf.choose_device(para['device'])
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
    for t in range(para['time_it']):
        states = pure_states_evolution(states, gate_list, which_where)
        if t % para['print_time'] == para['print_time']-1:
            list_states.append(states)
    states = tc.stack(list_states, dim=1)
    return states

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
    para['device'] = bf.choose_device(para['device'])

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
    for t in range(para['time_it']):
        # states = pure_states_evolution(states, [gate, gate_l, gate_r], which_where)
        gate_dot = tc.matrix_exp(- 1j * para['tau'] * H_dot).reshape([para['d']] * 6)
        states = pure_states_evolution(states, [gate_dot], [[0, 0, 1, 2]])

        for j in range(para['out_size']):
            site_pos = para['dot_size'] + j
            dot_pos = random.randint(0, para['dot_size']-1)

            gate_interact = tc.matrix_exp(- 1j * para['tau'] * para['alpha']**u[j] * SxSx).reshape([para['d']] * 4)
            pure_states_evolution(states, [gate_interact], [[0, dot_pos, site_pos]])

            gate_mag = tc.matrix_exp(- 1j * para['tau'] * h[j] * Sz).reshape([para['d']] * 2)
            pure_states_evolution(states, [gate_mag], [[0, site_pos]])

        if t % para['print_time'] == para['print_time']-1:
            list_states.append(states)
    states = tc.stack(list_states, dim=1)
    return states
