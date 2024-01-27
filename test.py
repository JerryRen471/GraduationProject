from cv2 import mean
import numpy as np
import torch as tc

# evol_mat = np.load('GraduationProject/Data/evol_mat.npy')
# evol_mat = tc.from_numpy(evol_mat)
# print('evol_mat.shape is', evol_mat.shape)

# A = tc.mm(evol_mat, evol_mat.T.conj())
# a = tc.diag(A)
# print(a)

def process_fide(U1:tc.Tensor, U0:tc.Tensor):
    M = tc.mm(U0.T.conj(), U1)
    n = U0.shape[0]
    f = 1/(n*(n+1)) * (n + tc.abs(tc.einsum('ii->',M))**2)
    return f

A = tc.rand([2,2], dtype=tc.complex128)
B = A - 1 * tc.rand([2,2], dtype=tc.complex128)
u, s, v = tc.svd(A)
U0 = tc.mm(u, v.T.conj())
u, s, v = tc.svd(B)
U1 = tc.mm(u, v.T.conj())
f = process_fide(U1=U0, U0=U0)
print(f)