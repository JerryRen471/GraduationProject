from matplotlib import pyplot as plt
import torch as tc
import numpy as np

file_path = "/data/home/scv7454/run/GraduationProject/Data/rand_unitary/n3/3body_1e-2noise_evol_mat.npy"
mat = np.load(file_path, allow_pickle=True)
print(mat.shape)
mat = tc.from_numpy(mat)
mat = mat.reshape(list(2 for _ in range(20)))
def half_bond_of_process(mpo):
    bond = None
    perm = []
    l = len(mpo.shape)//2
    for i in range(l):
        perm.append(i)
        perm.append(i+l)
    perm_mpo = tc.permute(mpo, perm)
    perm_mat = perm_mpo.reshape([2**(l//2*2), -1])
    bond = tc.linalg.svdvals(perm_mat)
    return bond

bond = half_bond_of_process(mat)
plt.plot(bond)
# print(bond)
def cut_bond(bond, cut_value):
    mask = (bond >= cut_value)
    bond = bond[mask]
    return bond

bond_cut = cut_bond(bond, 1)
print(bond_cut)