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

def entropy(lp):
    return tc.dot(-lp, tc.log2(lp))

# path_format = "/data/home/scv7454/run/GraduationProject/Data/rand_unitary/n{0}/{0}body_1e{1}noise_evol_mat.npy"

# for i in [1,2,3,10]:
#     plt.figure()
#     for j in [-3,-2,-1,0,1]:
#         bond = half_bond(path_format.format(i,j))
#         print(bond/tc.sum(bond))
#         print(tc.sum(bond))
#         ee = entropy(bond/tc.sum(bond))
#         # print(ee)
#         plt.plot(bond/tc.sum(bond), label="noise=1e{0:d},ee={1:.2f}".format(j, ee))
#     plt.title("{}body unitary gates".format(i))
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.legend()
#     plt.savefig("/data/home/scv7454/run/GraduationProject/pics/rand_unitary/n{}/bond_diff_noise.svg".format(i))
#     plt.close()
    
# print(bond)
def cut_bond(bond, cut_value):
    mask = (bond >= cut_value)
    bond = bond[mask]
    return bond

# bond_cut = cut_bond(bond, 1)
# print(bond_cut)



def main(file_path1):
    bond = half_bond(file_path1)
    print(tc.sum(bond))
    ee = entropy(bond/tc.sum(bond))
    return bond, ee
    
# bond, ee = main(file_path1)
# plt.plot(bond/tc.sum(bond), label="evol_time=0.01,ee={:.2f}".format(ee))
# bond, ee = main(file_path2)
# plt.plot(bond/tc.sum(bond), label="evol_time=0.01,ee={:.2f}".format(ee))

# plt.title("Heisenverg model")
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.savefig("/data/home/scv7454/run/GraduationProject/pics/Heis/bond_diff_evol_time.svg")
# plt.close()

if __name__ == '__main__':
    file_path1 = "/data/home/scv7454/run/GraduationProject/Data/Heis/evol_mat.npy"
    file_path2 = "/data/home/scv7454/run/GraduationProject/Data/Heis/evol_mat100.npy"

    path_format = "/data/home/scv7454/run/GraduationProject/Data/evol_mat{}.npy"
    e = list()
    for i in range(10, 110, 10):
        path = path_format.format(i)
        bond, ee = main(path)
        e.append(ee)
    print(e)
    plt.plot(e)
    plt.savefig("/data/home/scv7454/run/GraduationProject/pics/bond_diff_evol_time.svg")
    plt.close()

    gate_fidelity = [0.9998542,0.9987716,0.996651,0.9973619,0.9954584,0.9916964,0.9884063,0.9726967,0.9713727,0.9726504]
    ent = [0.3301,0.3837,0.5086,0.6147,0.7075,0.7905,0.8658,0.9353,1.0003,1.3142]
    plt.plot(ent, gate_fidelity)
    plt.savefig("/data/home/scv7454/run/GraduationProject/pics/fidelity_entropy.svg")