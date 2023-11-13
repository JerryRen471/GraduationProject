from cv2 import mean
import numpy as np
import torch as tc

# evol_mat = np.load('GraduationProject/Data/evol_mat.npy')
# evol_mat = tc.from_numpy(evol_mat)
# print('evol_mat.shape is', evol_mat.shape)

# A = tc.mm(evol_mat, evol_mat.T.conj())
# a = tc.diag(A)
# print(a)

def fidelity(psi1:tc.Tensor, psi0:tc.Tensor):
    psi0_ = psi0.reshape(psi0.shape[0], -1)
    psi1_ = psi1.reshape(psi1.shape[0], -1)
    fides = tc.einsum('ab,ab->a', psi1_.conj(), psi0_)
    f = tc.mean(fides)
    return f

from rand_dir_and_nondir import rand_states

psi0 = rand_states(16, 4, device=tc.device('cpu'))
psi1 = rand_states(16, 4, device=tc.device('cpu'))
print(fidelity(psi1, psi0))