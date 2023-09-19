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
    Evolve the state by several gates0.
    :param state: initial state
    :param gates: quantum gates
    :param which_where: [which gate, which spin, which spin]
    :return: evolved state
    Example: which_where = [[0, 1, 2], [1, 0, 1]] means gate 0 on spins
    1 and 2, and gate 1 on spins 0 and 1
    """

    def pure_state_evolution_one_gate(v, g, pos):
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


def time_evolution_Heisenberg_chain(para=None, state=None, save=True):
    para_def = dict()
    para_def['J'] = [0, 0, 1]
    para_def['h'] = [1.5, 0, 0]
    para_def['length'] = 18
    para_def['spin'] = 'half'
    para_def['BC'] = 'open'
    para_def['time_tot'] = 50*2
    para_def['tau'] = 0.01
    para_def['print_time_it'] = 10
    para_def['print_dtime'] = para_def['tau'] * para_def['print_time_it']
    para_def['hl'] = 2
    para_def['device'] = None
    para_def['dtype'] = tc.complex128
    # print(random.random())

    if para is None:
        para = dict()
    para = dict(para_def, **para)
    para['d'] = phy.from_spin2phys_dim(para['spin'])
    para['time_it'] = round(para['time_tot'] / para['tau'])
    # para['print_time_it'] = round(para['print_dtime'] / para['tau'])
    para['device'] = bf.choose_device(para['device'])
    # print(para)

    # 初态
    if state is None:
        state = tc.zeros((para['d'] ** para['length'], ),
                         device=para['device'], dtype=para['dtype'])
        state[0] = 1.0
        state = state.reshape([para['d']] * para['length'])

    hamilt = phy.hamiltonian_heisenberg(para['spin'], para['J'][0], para['J'][1], para['J'][2],
                                        [para['h'][0]/2, para['h'][0]/2],
                                        [para['h'][1]/2, para['h'][1]/2],
                                        [para['h'][2]/2, para['h'][2]/2],
                                        device=para['device'], dtype=para['dtype'])
    gate = tc.matrix_exp(- 1j * para['tau'] * hamilt).reshape([para['d']] * 4)
    # print(hamilt)
    # print(gate)

    hamilt_l = phy.hamiltonian_heisenberg(para['spin'], para['J'][0], para['J'][1], para['J'][2],
                                          [para['h'][0]+para['hl'], para['h'][0] / 2],
                                          [para['h'][1], para['h'][1] / 2],
                                          [para['h'][2], para['h'][2] / 2],
                                          device=para['device'], dtype=para['dtype'])
    gate_l = tc.matrix_exp(- 1j * para['tau'] * hamilt_l).reshape([para['d']] * 4)
    # print(hamilt_l)
    # print(gate_l)

    hamilt_r = phy.hamiltonian_heisenberg(para['spin'], para['J'][0], para['J'][1], para['J'][2],
                                          [para['h'][0] / 2, para['h'][0]],
                                          [para['h'][1] / 2, para['h'][1]],
                                          [para['h'][2] / 2, para['h'][2]],
                                          device=para['device'], dtype=para['dtype'])
    gate_r = tc.matrix_exp(- 1j * para['tau'] * hamilt_r).reshape([para['d']] * 4)
    # print(hamilt_r)
    # print(gate_r)

    # hamilt = tc.from_numpy(hamilt).to(para['device'])
    # hamilt_l = tc.from_numpy(hamilt_l).to(para['device'])
    # hamilt_r = tc.from_numpy(hamilt_r).to(para['device'])

    which_where = list()
    for n in range(para['length']-1):
        # [which gate, spin 1, spin 2]
        which_where.append([0, n, n+1])
    if para['BC'].lower() in ['pbc', 'periodic']:
        which_where.append([0, 0, para['length']-1])
    else:
        which_where[0][0] = 1
        which_where[-1][0] = 2

    spin = phy.spin_operators(para['spin'], device=para['device'])
    mag_z, eg, lm, ent = list(), list(), list(), list()
    states = list()
    for t in range(para['time_it']):
        if (t % para['print_time_it']) == 0:
            norm = tc.sqrt(tc.dot(state.reshape(-1, ), state.reshape(-1, ).conj()).real) # 修改
            state = state / norm # 修改
            states.append(state.cpu())
            mag_z.append(phy.magnetizations(state, [spin['sz']])) # 计算链上每个格点的z方向磁矩
            energies = phy.bond_energies(state, [hamilt.reshape([para['d']] * 4),
                                                 hamilt_l.reshape([para['d']] * 4),
                                                 hamilt_r.reshape([para['d']] * 4)], which_where)
            # eg_ = sum(energies) / state.ndimension()  # energy per site
            eg_ = sum(energies) / len(energies)  # energy per bond
            eg.append(eg_)
            lm_ = np.linalg.svd(state.reshape(2 ** round(para['length'] / 2), -1).to(
                'cpu').numpy(), compute_uv=False) # 计算以链中间为端点二分系统的奇异谱，对应二分纠缠谱
            lm.append(np.real(lm_))
            ent.append(phy.entanglement_entropy(lm_))
        state = pure_state_evolution(state, [gate, gate_l, gate_r], which_where)
        
        # print(state)
        # print(para)

    if save == True:
        print_field = tc.arange(0, len(ent)) * para['print_dtime']
        mag_z = tc.cat(mag_z, dim=0)
        mag_z_tot = mag_z.sum(dim=1) / mag_z.shape[1] # 不同时刻链的z方向总磁矩对链长做平均
        pf.plot(print_field, mag_z_tot)
        pf.plot(print_field, eg)
        pf.plot(print_field, ent)
        bf.output_txt(mag_z_tot, filename='GraduationProject/Data/'+para['folder']+'mag_z_dt{:d}.txt'.format(para['print_time_it']))
        np.save('GraduationProject/Data/'+para['folder']+'states_dt{:d}_tot{:.0f}'.format(para['print_time_it'], para['time_tot']), tc.stack(states))
    return state

if __name__ == '__main__':
    import time
    T1 = time.perf_counter()
    length = 10
    spin = 'half'
    d = phy.from_spin2phys_dim(spin)
    device = tc.device('cuda:0')
    dtype = tc.complex128

    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--Jx', type=float, default=1)
    parser.add_argument('--Jy', type=float, default=0)
    parser.add_argument('--Jz', type=float, default=1)
    parser.add_argument('--hx', type=float, default=0.5)
    parser.add_argument('--hl', type=float, default=2)
    parser.add_argument('--time', type=float, default=100)
    parser.add_argument('--pt_it', type=int, default=10)
    parser.add_argument('--folder', type=str, default='')
    args = parser.parse_args()
    n = args.n
    loc = 2 ** int(length-n) - 1

    para = dict()
    para['length'] = length
    para['spin'] = 'half'
    para['d'] = d
    para['device'] = device
    para['dtype'] = dtype
    para['J'] = [args.Jx, args.Jy, args.Jz]
    para['h'] = [args.hx, 0, 0]
    para['time_tot'] = args.time
    para['print_time_it'] = args.pt_it
    para['hl'] = args.hl
    para['folder'] = args.folder

    state = tc.zeros((d ** length, ), device=device, dtype=dtype)
    state[loc] = 1.0
    state = state.reshape([d] * length)
    # print(state)

    time_evolution_Heisenberg_chain(para, state)
    T2 = time.perf_counter()
    print('程序运行时间:%.4f秒'%((T2 - T1)))
