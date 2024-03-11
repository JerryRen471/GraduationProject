# from cv2 import mean
import numpy as np
import torch as tc

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

# print(qr_Haar(2, 2, device=tc.device('cpu')))

def Haar_rand_states(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [2 ** length]
    state1 = tc.zeros(shape, dtype=tc.complex128, device=device)
    state1[0] = 1
    print(state1)
    U = qr_Haar(number, length, device=device)
    states = tc.einsum('nij,j->ni', U, state1)
    return states

states = Haar_rand_states(2,2, device=tc.device('cpu'))
B = tc.einsum('ni,ni->n', states, states.conj())
print(B)