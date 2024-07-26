from copy import deepcopy
import torch as tc
from Library import PhysModule as phy
from Library.TensorNetwork import *
import time

device=tc.device('cpu')
dtype=tc.complex64
mps1 = rand_prod_mps_pack(4, 5, chi=4, phydim=2, device=device, dtype=dtype)
mps2 = rand_prod_mps_pack(4, 5, chi=4, phydim=2, device=device, dtype=dtype)
fidelity = inner_mps_pack(mps1, mps2)
print(fidelity)
sigma_z = tc.zeros([2, 2], device=device, dtype=dtype)
sigma_z[0, 0] = 1+0.j
sigma_z[1, 1] = -1+0.j
print(sigma_z)
mps_ = deepcopy(mps1)
mps_.node_list = mps1.node_list[:]
mps_.act_one_body_gate(sigma_z, 2)
print(tc.dist(mps_.node_list[2], mps1.node_list[2]))
hz = inner_mps_pack(mps1, mps_)
hz_ = multi_mags_from_mps_pack(mps1, spins=[sigma_z])
print(hz_.shape)
print(hz_)
# print(hz-hz_)
