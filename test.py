from cv2 import mean
import numpy as np
import torch as tc

# evol_mat = np.load('GraduationProject/Data/evol_mat.npy')
# evol_mat = tc.from_numpy(evol_mat)
# print('evol_mat.shape is', evol_mat.shape)

# A = tc.mm(evol_mat, evol_mat.T.conj())
# a = tc.diag(A)
# print(a)

'''用循环生成n个[1, m]的随机张量和直接生成一个[n, m]的随机张量是一样的'''

# tc.manual_seed(1)
def rand_normal_dist_states(number:int, length:int, device=tc.device('cuda:0'))->tc.Tensor:
    number = int(number)
    shape = [1, 2 ** length]
    states = tc.randn(shape, dtype=tc.complex128, device=device)
    sigma = tc.rand([number,1], dtype=tc.complex128, device=device)*1
    mu = tc.rand([number,1], dtype=tc.complex128, device=device)
    print('mu=',mu)
    states = states*sigma+mu
    shape_ = [number] + [2]*length
    norm = tc.sum(states * states.conj(), dim=1, keepdim=True)
    print('norm=',norm)
    # states = states / tc.sqrt(norm.real)
    # states = states.reshape(shape_)
    return states

a = rand_normal_dist_states(2, 10, device=tc.device('cpu'))
print('a=',a)
norm = tc.sum(a * a.conj(), dim=1, keepdim=True)
# print(norm)

b = tc.sum(a, dim=[1])/a.shape[1]
print(b)
print(tc.var(a, dim=1))

a = tc.tensor([[1, 2, 3], [4, 5, 6]])
b = tc.tensor([[1],[2]])
# print(a.shape)
# print(b.shape)
# print(a*b)

