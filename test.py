# from cv2 import mean
import numpy as np
import torch as tc

def gate_fidelity(E, U):
    n = E.shape[0]
    trace = tc.einsum('aa', U.T.conj() @ E)
    gate_fidelity = 1/(n*(n+1))*(n + tc.abs(trace)**2)
    return gate_fidelity

def similarity(E:tc.Tensor, U:tc.Tensor):
    '''
    E: circuit
    U: real process
    '''
    a = tc.norm(E - U)
    b = 2 * tc.norm(U)
    s = 1 - a/b
    return s

A = tc.rand([2,2], dtype=tc.complex128)
u, s, v = tc.svd(A)
U0 = tc.mm(u, v.T.conj())
f = gate_fidelity(U0, (0.5+0.866j)*U0)
s = similarity(U0, (0.5+0.866j)*U0)
print('gate_fidelity',f)
print('similarity', s)