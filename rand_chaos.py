from cProfile import label
from matplotlib import pyplot as plt
import torch as tc
import numpy as np
from Library.BasicFun import mkdir
from Library import BasicFun as bf
from Library import PhysModule as phy
import random

from rand_dir_and_nondir import random_uni_evl

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

'''用循环生成n个[1, m]的随机张量和直接生成一个[n, m]的随机张量是一样的'''
def rand_states_(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [number, 2 ** length]
    states = tc.randn(shape, dtype=tc.complex128, device=device)
    sigma = tc.rand([number,1], dtype=tc.float64, device=device)
    mu_real = (tc.rand([number,1], dtype=tc.float64, device=device)-0.5)*6
    mu_img = (tc.rand([number,1], dtype=tc.float64, device=device)-0.5)*6
    mu = tc.complex(mu_real, mu_img)
    # print('mu=',mu)
    states = states*sigma+mu
    shape_ = [number] + [2]*length
    norm = tc.sum(states * states.conj(), dim=1, keepdim=True)
    # print('norm=',norm)
    states = states / tc.sqrt(norm.real)
    states = states.reshape(shape_)
    return states

def qr_Haar(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [number, 2**length, 2**length]
    real = tc.randn(shape, dtype=tc.float64, device=device)
    imag = tc.randn(shape, dtype=tc.float64, device=device)
    A = tc.complex(real, imag)
    Q, R = tc.linalg.qr(A, 'complete')
    Lambda = tc.diagonal(R, dim1=1, dim2=2)/tc.abs(tc.diagonal(R, dim1=1, dim2=2))
    Haar_mat = tc.einsum('nij,nj->nij', Q, Lambda)
    return Haar_mat

def Haar_rand_states(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [2 ** length]
    state1 = tc.zeros(shape, dtype=tc.complex128, device=device)
    state1[0] = 1
    print(state1)
    U = qr_Haar(number, length, device=device)
    states = tc.einsum('nij,j->ni', U, state1)
    shape_ = [number] + [2]*length
    states = states.reshape(shape_)
    return states

def Haar_random_product_states(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    product_states = Haar_rand_states(number, 1, device=device)
    for _ in range(length-1):
        temp = Haar_rand_states(number, 1, device=device)
        product_states = tc.einsum("n...i,nj->n...ij", product_states, temp)
    return product_states

def rand_states(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [number, 2 ** length]
    states = tc.randn(shape, dtype=tc.complex128, device=device)
    shape_ = [number] + [2]*length
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

def gen_select(gen_type:str):
    if gen_type == 'd':
        gen = rand_dir_prod_states
    elif gen_type == 'n':
        gen = rand_states
    elif gen_type == 'h':
        gen = Haar_random_product_states
    # else:
    #     gen = None
    #     print('-'*10+'\nArgument --gen_type must be the combination of \'n\' and \'d\'!\n'+'-'*10)
    return gen

def label_generate(data, para):
    label = XXZ_inhomo_mul_states_evl(data, para)
    input = tc.cat((data.unsqueeze(1), label[:, :-1]), dim=1)
    merge = lambda x: x.reshape(x.shape[0]*x.shape[1], *x.shape[2:])
    return merge(input), merge(label)

if __name__ == '__main__':
    # length = 14
    spin = 'half'
    d = phy.from_spin2phys_dim(spin)
    device = tc.device('cuda:0')
    dtype = tc.complex128

    import argparse
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--J', type=float, default=1)
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('--theta', type=float, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--evol_num', type=int, default=10)
    parser.add_argument('--test_num', type=int, default=500)
    parser.add_argument('--folder', type=str, default='mixed_rand_states/')
    parser.add_argument('--evol_mat_path', type=str, default="GraduationProject/Data/evol_mat.npy")
    parser.add_argument('--gen_type', type=str, default='nn')
    parser.add_argument('--time_interval', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.01)
    args = parser.parse_args()

    para = dict()
    para['length'] = args.length
    para['device'] = device
    para['gen_type'] = args.gen_type
    para['spin'] = 'half'
    para['d'] = d
    para['dtype'] = dtype
    para['J'] = args.J
    para['delta'] = args.delta
    para['theta'] = args.theta

    if args.seed != None:
        tc.manual_seed(args.seed)

    path = 'GraduationProject/Data'+args.folder
    mkdir(path)
    para_evol = dict(para)
    para_evol['time_tot'] = args.time_interval
    para_evol['print_time'] = args.time_interval

    para_train = dict(para)
    para_train['time_tot'] = args.time_interval*args.evol_num
    para_train['print_time'] = args.time_interval
    para_train['sample_num'] = args.sample_num

    para_test = dict(para)
    para_test['time_tot'] = args.time_interval
    para_test['print_time'] = args.time_interval
    para_test['sample_num'] = args.sample_num

    import os
    evol_mat_path = args.evol_mat_path
    # if os.access(evol_mat_path, os.F_OK):
    #     pass
    # else:
    E = tc.eye(2**para['length'], dtype=tc.complex128, device=para['device'])
    shape_ = [E.shape[0]] + [2] * para['length']
    E = E.reshape(shape_)
    evol_mat = XXZ_inhomo_mul_states_evl(E, para_evol).reshape(E.shape[0], -1)
    print('evol_mat.shape is', evol_mat.shape)
    np.save(evol_mat_path, evol_mat.cpu().T)

    gen_train = gen_select(para['gen_type'][0])
    trainset = gen_train(args.sample_num, para['length'], device=para['device'])
    trainset, train_lbs = label_generate(trainset, para_train)

    gen_test = gen_select(para['gen_type'][1])
    testset = gen_train(1, para['length'], device=para['device'])
    testset, test_lbs = label_generate(testset, para_test)
    data = dict()
    data['train_set'] = trainset
    data['train_lbs'] = train_lbs
    data['test_set'] = testset
    data['test_lbs'] = test_lbs
    
    np.save(path+'/train_set_sample_{:d}_evol_{:d}'.format(args.sample_num, args.evol_num), data)

    from Library.PhysModule import multi_mags_from_states
    multi_mags = multi_mags_from_states(trainset)
    print(multi_mags.shape)
    s_label = 'xyz'
    for i in range(multi_mags.shape[2]):
        mkdir('GraduationProject/pics'+args.folder+f'/mag_evol{args.evol_num}')
        legend = []
        time_it = multi_mags.shape[0]
        x = np.array([_ for _ in range(time_it)]) * args.time_interval
        for j in range(3):
            y = np.array(multi_mags.cpu()[:, j, i])
            plt.plot(x, y, label='s'+s_label[j])
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('si')
        plt.title(f'Average magnetic moment on site {i}')
        plt.savefig('GraduationProject/pics'+args.folder+f'/mag_evol{args.evol_num}/AverageMagneticMoment{i}.svg')
        plt.close()