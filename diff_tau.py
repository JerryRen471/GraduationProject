from cgitb import small
import torch as tc
import numpy as np

from model_evol import PXP_mul_states_evl
from gen_init import *

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--length', type=int, default=10)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--evol_num', type=int, default=1)
parser.add_argument('--test_num', type=int, default=500)
parser.add_argument('--entangle_dim', type=int, default=1)
parser.add_argument('--folder', type=str, default='mixed_rand_states/')
parser.add_argument('--evol_mat_path', type=str, default="GraduationProject/Data/evol_mat.npy")
parser.add_argument('--gen_type', type=str, default='d')
parser.add_argument('--time_interval', type=float, default=0.02)
parser.add_argument('--tau', type=float, default=0.02)
args = parser.parse_args()

device = tc.device('cpu')
dtype = tc.complex128

para = dict()
para['length'] = args.length
para['device'] = device
para['gen_type'] = args.gen_type
para['spin'] = 'half'
para['d'] = 2
para['dtype'] = dtype
para['tau'] = args.tau

para_train = dict(para)
para_train['time_tot'] = args.time_interval*args.evol_num
para_train['print_time'] = args.time_interval
para_train['sample_num'] = args.sample_num

para_small_tau = dict(para_train)
para_small_tau['tau'] = 0.02
para_big_tau = dict(para_train)
para_big_tau['tau'] = 0.002

init_states = rand_dir_prod_states(para_train['sample_num'], para['length'], device=para['device'], dtype=para['dtype'], entangle_dim=args.entangle_dim)
small_tau_evol = PXP_mul_states_evl(init_states, para_small_tau)
big_tau_evol = PXP_mul_states_evl(init_states, para_big_tau)

print(tc.dist(small_tau_evol, big_tau_evol))