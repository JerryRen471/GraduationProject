# from cv2 import mean
import numpy as np
import torch as tc

A = tc.randn([2,2,3,3], dtype=tc.complex64)
Q, R = tc.linalg.qr(A)
print(Q.shape)
# Q = Q.reshape([2,3,3])
Q1 = Q[0,0]
Q2 = Q[0,1]
print(tc.mm(Q1, Q1.T.conj()))
print(tc.mm(Q2, Q2.T.conj()))
print(tc.mm(Q1.T.conj(), Q1))
print(tc.mm(Q2.T.conj(), Q2))
# print(tc.mm(Q, Q.T.conj()))
# print(tc.mm(Q.T.conj(), Q))