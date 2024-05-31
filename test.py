from matplotlib import pyplot as plt
import torch as tc
import numpy as np

def spectrum(mat:tc.Tensor):
    energy = tc.log(tc.linalg.eigvals(mat))/1.j
    energy = energy.real
    energy, ind = tc.sort(energy)
    return energy

train_num = 10
folder = '/loss_multi_mags/0.01/dn'
data_path = 'GraduationProject/Data'+folder
pic_path = 'GraduationProject/pics'+folder
evol_mat_path = 'GraduationProject/Data/evol_mat1.npy'
qc_mat = np.load(data_path+'/qc_mat_num{:d}.npy'.format(train_num))
qc_mat = tc.from_numpy(qc_mat)
evol_mat = np.load(evol_mat_path)
evol_mat = tc.from_numpy(evol_mat)
qc_energy = spectrum(qc_mat)
evol_energy = spectrum(evol_mat)

legends = []
plt.plot(qc_energy, label='qc')
plt.plot(evol_energy, label='Trotter')
plt.legend()
plt.xlabel('n')
plt.ylabel('E*t')
plt.savefig(pic_path+'/spectrum_num{:d}.svg'.format(train_num))
plt.close()