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

def Heisenberg_mul_states_evl(states, para=None):
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
    para_def['dtype'] = tc.complex64
    if para is None:
        para = dict()
    para = dict(para_def, **para)
    para['d'] = phy.from_spin2phys_dim(para['spin'])
    para['time_it'] = round(para['time_tot'] / para['tau'])
    # para['print_time_it'] = round(para['print_dtime'] / para['tau'])
    para['device'] = bf.choose_device(para['device'])
    # print(para)

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

    which_where = list()
    for n in range(para['length']-1):
        # [which gate, spin 1, spin 2]
        which_where.append([0, n, n+1])
    if para['BC'].lower() in ['pbc', 'periodic']:
        which_where.append([0, 0, para['length']-1])
    else:
        which_where[0][0] = 1
        which_where[-1][0] = 2
    states_new = pure_states_evolution(states, [gate, gate_l, gate_r], which_where)
    return states_new

def rand_states(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [number, 2 ** length]
    states = tc.rand(shape, dtype=tc.complex128, device=device)
    shape_ = [number] + [2]*10
    norm = tc.sum(states * states.conj(), dim=1, keepdim=True)
    states = states / tc.sqrt(norm)
    states = states.reshape(shape_)
    return states

def rand_dir_prod_states(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [number, 2]
    states = tc.rand(shape, dtype=tc.complex128, device=device)
    for _ in range(length - 1):
        states = tc.einsum('ij,ik->ijk', states, tc.rand(shape, dtype=tc.complex128, device=device))
        states = states.reshape(number, -1)
    print(states.shape)
    shape_ = [number] + [2]*length
    norm = tc.sum(states * states.conj(), dim=1, keepdim=True)
    states = states / tc.sqrt(norm)
    states = states.reshape(shape_)
    return states

def gen_select(gen_type:str)->function:
    if gen_type == 'd':
        gen = rand_dir_prod_states
    elif gen_type == 'n':
        gen = rand_states
    else:
        gen = None
        print('-'*10+'\nArgument --gen_type must be the combination of \'n\' and \'d\'!\n'+'-'*10)
    return gen

if __name__ == 'main':
    length = 10
    spin = 'half'
    d = phy.from_spin2phys_dim(spin)
    device = tc.device('cuda:0')
    dtype = tc.complex128
    seed = 100

    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--Jx', type=float, default=1)
    parser.add_argument('--Jy', type=float, default=0)
    parser.add_argument('--Jz', type=float, default=1)
    parser.add_argument('--hx', type=float, default=0.5)
    parser.add_argument('--hl', type=float, default=2)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--train_num', type=int, default=100)
    parser.add_argument('--test_num', type=int, default=500)
    parser.add_argument('--folder', type=str, default='mixed_rand_states/')
    parser.add_argument('--gen_type', type=str, default='nn')
    args = parser.parse_args()

    para = dict()
    para['length'] = length
    para['spin'] = 'half'
    para['d'] = d
    para['device'] = device
    para['dtype'] = dtype
    para['J'] = [args.Jx, args.Jy, args.Jz]
    para['h'] = [args.hx, 0, 0]
    para['time_tot'] = 0.01
    para['print_time_it'] = 1
    para['hl'] = args.hl
    para['folder'] = args.folder
    para['seed'] = args.seed

    if para['seed'] != None:
        tc.manual_seed(para['seed'])
    gen_train = gen_select(args.gen_type[0])
    trainset = gen_train(args.train_num, length, device=para['device'])
    print(trainset.dtype)
    train_lbs = Heisenberg_mul_states_evl(trainset, para)

    gen_test = gen_select(args.gen_type[1])
    testset = rand_states(args.test_num, length, device=para['device'])
    test_lbs = Heisenberg_mul_states_evl(testset, para)
    data = dict()
    data['train_set'] = trainset
    data['train_lbs'] = train_lbs
    data['test_set'] = testset
    data['test_lbs'] = test_lbs

    np.save('GraduationProject/Data/'+para['folder']+'data_num{:d}'.format(args.train_num), data)