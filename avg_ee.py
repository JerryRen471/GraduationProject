from matplotlib import pyplot as plt
import torch as tc
import numpy as np
from rand_dir_and_nondir import rand_dir_prod_states, random_uni_evl, Haar_random_product_states
from draw_bond import main

def avg_ee_from_path(states, file_path):
    mat = np.load(file_path, allow_pickle=True)
    # print(mat.shape)
    mat = tc.from_numpy(mat)
    states_ = random_uni_evl(states, mat)
    shape = [num, int(2**(length/2)), int(2**(length/2))]
    states_ = states_.reshape(shape)
    # print(states_)
    spectral = tc.linalg.svdvals(states_)
    p_distribution = spectral**2
    # print(p_distribution.shape)
    entropy = tc.einsum("ij,ij->i", -p_distribution, tc.log2(p_distribution))
    avg_ee = tc.sum(entropy)/entropy.shape[0]
    return entropy, avg_ee

num = 10000
length = 10
device = tc.device('cpu')
states = Haar_random_product_states(num, length, device)

path_format = "/data/home/scv7454/run/GraduationProject/Data/evol_mat{}.npy"

avg_ee_list = []
e = []
for i in range(10, 110, 10):
    path = path_format.format(i)
    ee, avg_ee = avg_ee_from_path(states, path)
    avg_ee_list.append(avg_ee)
    bond, ee = main(path)
    e.append(ee)
print(avg_ee_list)
print(e)
plt.plot(avg_ee_list)
plt.savefig("/data/home/scv7454/run/GraduationProject/pics/10000rand_states_entropy_diff_evol_time.svg")
plt.close()

plt.plot(avg_ee_list, e)
plt.xlabel('average_ee')
plt.ylabel('bond_entropy')
plt.savefig("/data/home/scv7454/run/GraduationProject/pics/10000avgVSbond.svg")
plt.close()