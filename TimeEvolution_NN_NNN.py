import sys
sys.path.append(sys.path[0]+'\\Library')
print(sys.path)

import torch as tc
import numpy as np
from Library import BasicFun as bf
from Library import PlotFun as pf
from Library import PhysModule as phy
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pure_state_evolution(state, gates, which_where):
    """
    :param state: initial state
    :param gates: quantum gates
    :param which_where: [which gate, which spin, which spin]
    :return: evolved state

    Evolve the state by several gates0.

    Example: which_where = [[0, 1, 2], [1, 0, 1]] means gate 0 on spins
    1 and 2, and gate 1 on spins 0 and 1
    """

    def pure_state_evolution_one_gate(v, g, pos):
        '''
        :param pos: g在v上作用的位置
        '''
        ind = list(range(len(pos), 2*len(pos)))
        v = tc.tensordot(v, g, [pos, ind])
        ind0 = list(range(v.ndimension()))
        for nn in range(len(pos)):
            ind0.remove(pos[nn])
        ind0 += pos
        order = np.argsort(ind0)
        return v.permute(tuple(order))

    for n in range(len(which_where)):
        state = pure_state_evolution_one_gate(
            state, gates[which_where[n][0]], which_where[n][1:])
    return state


def time_evolution_NNN_chain(para=None, state=None):
    para_def = dict()
    para_def['t'] = [1, 0]
    para_def['V'] = [1, 0]
    para_def['length'] = 18
    para_def['spin'] = 'half'
    para_def['BC'] = 'pbc'
    para_def['time_tot'] = 50*2
    para_def['tau'] = 0.01
    para_def['print_dtime'] = 1
    para_def['device'] = None
    para_def['dtype'] = tc.complex128
    # print(random.random())

    if para is None:
        para = dict()
    para = dict(para_def, **para)
    para['d'] = phy.from_spin2phys_dim(para['spin'])
    para['time_it'] = round(para['time_tot'] / para['tau'])
    para['print_time_it'] = round(para['print_dtime'] / para['tau'])
    para['device'] = bf.choose_device(para['device'])
    print(para)

    # 初态
    if state is None:
        state = tc.zeros((para['d'] ** para['length'], ),
                         device=para['device'], dtype=para['dtype'])
        state[0] = 1.0
        state = state.reshape([para['d']] * para['length'])

    hamilt = phy.hamiltonian_NN_NNN(para['spin'], para['t'][0], para['t'][1], para['V'][0], para['V'][1],
                                        device=para['device'], dtype=para['dtype'])
    gate = tc.matrix_exp(- 1j * para['tau'] * hamilt).reshape([para['d']] * 6)

    which_where = list()
    for n in range(para['length']-2):
        # [which gate, spin 1, spin 2, spin 3]
        which_where.append([0, n, n+1, n+2])
    if para['BC'].lower() in ['pbc', 'periodic']:
        which_where.append([0, para['length']-2, para['length']-1, 0])
        which_where.append([0, para['length']-1, 0, 1])
    # else:
    #     which_where[0][0] = 1
    #     which_where[-1][0] = 2

    spin = phy.spin_operators(para['spin'], device=para['device'])
    mag_z, eg, lm, ent = list(), list(), list(), list()
    sigma_xx = list()
    for t in range(para['time_it']):
        if (t % para['print_time_it']) == 0:
            norm = tc.sqrt(tc.dot(state.reshape(-1, ), state.reshape(-1, ).conj()).real)
            state = state / norm
            mag_z.append(phy.combined_mag(state, [spin['sz']])) # 计算链上每个格点的z方向磁矩
            energies = phy.bond_energies(state, [hamilt.reshape([para['d']] * 6)], which_where)
            # eg_ = sum(energies) / state.ndimension()  # energy per site
            eg_ = sum(energies) / len(energies)  # energy per bond
            eg.append(eg_)
            lm_ = np.linalg.svd(state.reshape(2 ** round(para['length'] / 2), -1).to(
                'cpu').numpy(), compute_uv=False) # 计算以链中间为端点二分系统的奇异谱，对应二分纠缠谱
            lm.append(np.real(lm_))
            ent.append(phy.entanglement_entropy(lm_))
            print('t={}'.format(t))
        state = pure_state_evolution(state, [gate], which_where)
        # print(state)
        # print(para)

    print_field = tc.arange(0, len(ent)) * para['print_dtime']
    mag_z = tc.cat(mag_z, dim=0)
    mag_z_0 = mag_z[:, 9]
    mag_z_tot = mag_z.sum(dim=1) / mag_z.shape[1] # 不同时刻链的z方向总磁矩对链长做平均
    pf.plot(print_field, mag_z_tot)
    pf.plot(print_field, eg)
    pf.plot(print_field, ent)
    bf.output_txt(mag_z_0, filename='D:\\Manual\\Code\\tests\\Graduation Project\\mag_z_0.txt')
    bf.output_txt(mag_z_tot, filename='D:\\Manual\\Code\\tests\\Graduation Project\\data.txt')
    bf.output_txt(eg, filename='D:\\Manual\\Code\\tests\\Graduation Project\\eg_data.txt')
    bf.output_txt(ent, filename='D:\\Manual\\Code\\tests\\Graduation Project\\ent_data.txt')
    return state

if __name__ == '__main__':
    # import time
    # T1 = time.perf_counter()
    length = 18
    spin = 'half'
    d = phy.from_spin2phys_dim(spin)
    device = tc.device('cuda:0')
    dtype = tc.complex128

    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--t1', type=float, default=1)
    parser.add_argument('--t2', type=float, default=0.1)
    parser.add_argument('--V1', type=float, default=1)
    parser.add_argument('--V2', type=float, default=0.1)
    args = parser.parse_args()
    n = args.n
    loc = 2 ** int(length-n) - 1

    para = dict()
    para['length'] = length
    para['spin'] = 'half'
    para['d'] = d
    para['device'] = device
    para['dtype'] = dtype
    para['t'] = [args.t1, args.t2]
    para['V'] = [args.V1, args.V2]

    state = tc.zeros((d ** length, ), device=device, dtype=dtype)
    state[loc] = 1.0
    state = state.reshape([d] * length)
    print(state.shape)
    
    time_evolution_NNN_chain(para, state)
    # T2 = time.perf_counter()
    # print('程序运行时间:%.4f秒'%((T2 - T1)))
