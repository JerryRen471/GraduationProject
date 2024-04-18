from matplotlib import pyplot as plt
import torch as tc
import numpy as np

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

def half_bond(file_path):
    mat = np.load(file_path, allow_pickle=True)
    print(mat.shape)
    mat = tc.from_numpy(mat)
    mat = mat.reshape(list(2 for _ in range(20)))
    bond = half_bond_of_process(mat)
    return bond

path_format = "/data/home/scv7454/run/GraduationProject/Data/rand_unitary/n{0}/{0}body_1e{1}noise_evol_mat.npy"

for i in [1,2,3,10]:
    plt.figure()
    for j in [-3,-2,-1,0,1]:
        bond = half_bond(path_format.format(i,j))
        plt.plot(bond, label="noise=1e{}".format(j))
    plt.title("{}body unitary gates".format(i))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig("/data/home/scv7454/run/GraduationProject/pics/rand_unitary/n{}/bond_diff_noise.svg".format(i))
    plt.close()
    
# print(bond)
def cut_bond(bond, cut_value):
    mask = (bond >= cut_value)
    bond = bond[mask]
    return bond

# bond_cut = cut_bond(bond, 1)
# print(bond_cut)


