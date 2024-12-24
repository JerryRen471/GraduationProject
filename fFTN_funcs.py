import torch as tc
import torch.nn as nn
import copy
import scipy as sc
import math as mt
import time
import matplotlib.pyplot as plt
import math
from epsilon_dec import levi_civita_nd, tt_svd
from tqdm import tqdm
import numpy as np
import itertools
import torch.optim as optim
import copy
from scipy.special import hermite
from scipy.integrate import quad

#tc.autograd.set_detect_anomaly(True)

def hars(x, n):
    return hermite(n)(x) * np.exp(-x**2 / 2)

def three_body_integrate_hermite(m, n, p):
    result, _ = quad(lambda x: hars(x, m) * hars(x, n) * hars(x, p), -np.inf, np.inf)
    result *= (1/(np.sqrt(np.sqrt(np.pi) * 2**m * np.math.factorial(m)) * np.sqrt(np.sqrt(np.pi) * 2**n * np.math.factorial(n)) * np.sqrt(np.sqrt(np.pi) * 2**p * np.math.factorial(p))))
    return result

def transfer_U(d, basis, dtype, device):
    if basis == 'harmonic':
        U = tc.zeros((d,d,d), dtype=dtype, device=device)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    U[i,j,k] = three_body_integrate_hermite(i,j,k)
        return U
    
def basis_transfer(U_transfer, d, chi, list):
    list_new_mps = []
    v = tc.ones((1,1,1), dtype = list[0][0].dtype, device = list[0][0].device)
    for i in range(len(list)):
        T = list[0][i]
        for j in range(len(list)-1): 
            a, b, c = T.shape   
            T = tc.einsum('abc,ijk,bjt->aitck', T, list[j+1][i], U_transfer).reshape(a*chi,b,c*chi)
        list_new_mps.append(T)
    L = list[0][-1]
    for k in range(len(list)-1):
        #print("L的shape", L.shape)
        p,s,q = L.shape
        L = tc.einsum('abc,ijk, bjt->aitck', L, list[k+1][-1], v).reshape(p*chi,s,q*chi)
        #print("相乘后L的shape", L.shape)
    list_new_mps.append(L)
    return list_new_mps
    
def choose_device(n):
    if n == 'cuda' and tc.cuda.is_available():
        dev = tc.device('cuda')
    else:
        dev = tc.device('cpu')
    return dev

def random_mps(length, d, chi, boundary='open', device=None, dtype=tc.float64):
    device = choose_device(device)
    if boundary == 'open':
        tensors = [tc.randn((chi, d, chi), device=device, dtype=dtype, requires_grad=True)
                   for _ in range(length - 2)]
        return [tc.randn((1, d, chi), device=device, dtype=dtype, requires_grad=True)] + tensors + [
            tc.randn((chi, d, 1), device=device, dtype=dtype, requires_grad=True)]
    else:  # 周期边界MPS
        return [tc.randn((chi, d, chi), device=device, dtype=dtype, requires_grad=True)
                for _ in range(length)]

def mps_init(paras, requires_grad):
    mps = nn.ParameterList(random_mps(paras['length'], paras['d'], paras['chi'], paras['boundary'], paras['device'], paras['dtype']))
    return mps

def orth_l2r(listp, dcut = -1, normalize = False):
    for i in range(len(listp)):
        #print(f"-------------------开始处理第{i+1}个张量-------------------")
        sp = listp[i].shape
        #print(f"第{i+1}个张量的形状是：", sp)
        if 0<dcut<sp[-1]:
            if_turn = True
        else:
            if_turn = False

        dec_tensor = listp[i].reshape(-1, sp[-1])
        p,q = dec_tensor.shape
        ra = tc.rand((p,q), dtype=tc.float64, device=tc.device('cpu')) * 1e-5
        dec_tensor = dec_tensor + ra
        #print(f"第{i+1}个张量的展开形状是：", dec_tensor.shape)
        u, s, v = tc.linalg.svd(dec_tensor, full_matrices=False)
        s = s
        print(f"第{i+1}个张量的SVD给出的奇异谱是：", s)
        if if_turn:
            #print("需要截断")
            u1 = u[:, :dcut]
            s1 = s[:dcut]
            v1 = v[:dcut, :]
            #print(f"截断后的U，S，V的形状是：{u.shape, s.shape, v.shape}")
            vslash = tc.diag(s1).mm(v1)
        else:
            #print("不需要截断")
            u1 = u
            vslash = tc.diag(s).mm(v)
        listp[i] = u1.reshape(sp[0], sp[1], -1)
        #print(f"第{i+1}个张量的重新reshape后的形状是：", listp[i].shape)
        if normalize:
            vslash = vslash / tc.norm(vslash)
        if i < len(listp)-1:
           #print("没有截断最后一个tensor")
           listp[i+1] = tc.tensordot(vslash, listp[i+1], ([1], [0]))
           #print(f"第{i+2}个张量的点乘后的形状是：", listp[i+1].shape)
        else:
           #print("截断最后一个tensor")
           listp[0] = tc.tensordot(vslash, listp[0], ([1], [0]))
           #print(f"第{1}个张量的点乘后的形状是：", listp[0].shape)
    return listp

def extend_phys_dim_by_value(mps, dim_add, bond, value=0):
    shape = list(mps[bond].shape)
    shape[1] = dim_add
    tensor_ = tc.ones(shape) * value
    mps[bond] = tc.cat([mps[bond], tensor_], dim=1)
    return mps

def match_phys_dims(mps1, mps2):
    assert len(mps1) == len(mps2)
    for n in range(len(mps1)):
        d1, d2 = mps1[n].shape[1], mps2[n].shape[1]
        if d1 < d2:
            mps1 = extend_phys_dim_by_value(mps1, d2 - d1, n, value=0)
        elif d1 > d2:
            mps2 = extend_phys_dim_by_value(mps2, d1 - d2, n, value=0)
    return mps1, mps2


def add_two_mps(mps1, mps2, bondary = 'par', auto_extend=True):
    mps = list()
    dev = mps1[0].device
    dt = mps1[0].dtype
    if auto_extend:
        mps1, mps2 = match_phys_dims(mps1, mps2)
    for n in range(len(mps1)):
        chi1L, d1, chi1R = mps1[n].shape
        #print(f"第{n+1}个张量的形状是：", mps1[n].shape)
        chi2L, d2, chi2R = mps2[n].shape
        #print(f"第{n+1}个张量的形状是：", mps2[n].shape)
        if bondary == 'open':
            if n == 0:
                tensor = tc.zeros((chi1L + chi2L, d1, chi1R + chi2R), device=dev, dtype=dt )
                tensor[:, :, :chi1R] = mps1[n]
                tensor[:, :, chi1R:chi1R + chi2R] = mps2[n]
            elif n == len(mps1) - 1:
                tensor = tc.zeros((chi1L + chi2L, d1, chi1R + chi2R), device=dev, dtype=dt )
                tensor[:chi1L, :, :] = mps1[n]
                tensor[chi1L:chi1L + chi2L, :, :] = mps2[n]
            else:
                tensor = tc.zeros((chi1L + chi2L, d1, chi1R + chi2R), device=dev, dtype=dt )
                tensor[:chi1L, :, :chi1R] = mps1[n]
                tensor[chi1L:chi1L + chi2L, :, chi1R:chi1R + chi2R] = mps2[n]
        else:
            tensor = tc.zeros((chi1L + chi2L, d1, chi1R + chi2R), device=dev, dtype=dt )
            tensor[:chi1L, :, :chi1R] = mps1[n]
            tensor[chi1L:chi1L + chi2L, :, chi1R:chi1R + chi2R] = mps2[n]
        mps.append(tensor)
    return mps

def delta_tensor_init(dim1,dim2,dim3, device=None, dtype=tc.float64):
    zeros = tc.zeros((dim1, dim2, dim3), dtype=dtype, device=device)
    ones = tc.zeros_like(zeros)
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                if i == j and j == k:
                    ones[i, j, k] = 1
    return ones

def B_init(l, d, chi, dtype, device):
    B_matrix = [tc.rand((chi,d,chi), dtype = dtype, device = device, requires_grad = True) for _ in range(l)]
    lambda_matrix = [tc.rand((d,1), dtype = dtype, device = device, requires_grad = True) for _ in range(l)]
    B_list = nn.ParameterList(B_matrix)
    lambda_list = nn.ParameterList(lambda_matrix)
    return B_list, lambda_list

def det_ansatz(list):
    tensors = [list[0], list[1], list[2]]
    lambda_list = [list[3], list[4], list[5]]
    a,b,c = tensors[0].shape
    device = tensors[0].device
    dtype = tensors[0].dtype
    length = len(tensors)
    each_mps = [delta_tensor_init(a,b,c,device=device,dtype=dtype) for _ in range(length+1)]
    #每一行的MPS:$\phi_i(x_row)$
    rows_mps = [copy.deepcopy(each_mps) for _ in range(length)]
    #按行排列的所有对称矩阵形式：所有行
    det_az = [copy.deepcopy(rows_mps) for _ in range(length)] #所有的N^2个MPS，其中每个子列表添加一行MPS
    for i in range(length):
        for j in range(length):
            det_az[i][j][j] = tensors[i].clone()
            det_az[i][j][-1] = tc.einsum('abc,bd->adc', det_az[i][j][-1], lambda_list[i].clone())
            #print(f"行列式的第{i+1}行第{j+1}列的第{j+1}个delta tensor被替换成B{i+1} tenosr")
    return det_az

def three_mps_transfer(list_det, U, d, chi, basis, dtype, device):
    list_mps1 = [list_det[0][0], list_det[1][1], list_det[2][2]]
    list_new1 = basis_transfer(U, d, chi, list_mps1)
    #print("list_new1的shape", list_new1[0].shape, list_new1[1].shape, list_new1[2].shape, list_new1[3].shape)
    list_mps2 = [list_det[0][1], list_det[1][2], list_det[2][0]]
    list_new2 = basis_transfer(U, d, chi, list_mps2)
    #print("list_new2的shape", list_new2[0].shape, list_new2[1].shape, list_new2[2].shape, list_new2[3].shape)
    list_mps3 = [list_det[0][2], list_det[1][0], list_det[2][1]]
    list_new3 = basis_transfer(U, d, chi, list_mps3)
    #print("list_new3的shape", list_new3[0].shape, list_new3[1].shape, list_new3[2].shape, list_new3[3].shape)
    list_mps_1 = [list_det[0][0], list_det[1][2], list_det[2][1]]
    list_new_1 = basis_transfer(U, d, chi, list_mps_1)
    for i in range(3):
        list_new_1[i] = -1.0 * list_new_1[i]
    #print("list_new_1的shape", list_new_1[0].shape, list_new_1[1].shape, list_new_1[2].shape, list_new_1[3].shape)
    list_mps_2 = [list_det[0][1], list_det[1][0], list_det[2][2]]
    list_new_2 = basis_transfer(U, d, chi, list_mps_2)
    for i in range(3):
        list_new_2[i] = -1.0 * list_new_2[i]
    #print("list_new_2的shape", list_new_2[0].shape, list_new_2[1].shape, list_new_2[2].shape, list_new_2[3].shape)
    list_mps_3 = [list_det[0][2], list_det[1][1], list_det[2][0]]
    list_new_3 = basis_transfer(U, d, chi, list_mps_3)
    for i in range(3):
        list_new_3[i] = -1.0 * list_new_3[i]
    #print("list_new_3的shape", list_new_3[0].shape, list_new_3[1].shape, list_new_3[2].shape, list_new_3[3].shape)
    list_mps = add_two_mps(list_new1, list_new2, bondary = 'par', auto_extend=False)
    list_mps = add_two_mps(list_mps, list_new3, bondary = 'par', auto_extend=False)
    list_mps = add_two_mps(list_mps, list_new_1, bondary = 'par', auto_extend=False)
    list_mps = add_two_mps(list_mps, list_new_2, bondary = 'par', auto_extend=False)
    list_mps = add_two_mps(list_mps, list_new_3, bondary = 'par', auto_extend=False)
    return list_mps


def clone_mps(mps):
    return [x.clone() for x in mps]

def normalize_mps_par(mps, paras):
    mps0 = clone_mps(mps) 
    v = tc.eye(mps[0].shape[0]**2, dtype=tc.float64, device=paras['device']).reshape([mps[0].shape[0]] * 4)
    #v = tc.ones((mps[0].shape[0], mps[0].shape[0]), dtype=tc.float64, device=paras['device'])
    mps_n = []
    norm = tc.zeros(len(mps)+1, dtype=tc.float64, device=paras['device'])
    for i in range(len(mps)):
        v = tc.einsum('uvab,acd,bce->uvde', v, mps0[i].conj(), mps0[i])
        #v = tc.einsum('ab,acd,bce->de', v, mps0[i].conj(), mps0[i])
        #n = tc.einsum('abcd->', v)
        n = tc.norm(v)
        mps_n.append(mps[i] / tc.sqrt(n))
        v = v / n
        norm[i] = n
    if v.numel() > 1:
        n = tc.einsum('acac->', v)
        mps_n[-1] = (mps_n[-1] / tc.sqrt(n))
        norm[-1] = n
    norms = tc.prod(norm)
    return norms, mps_n

def inner_product(tensors0, tensors1, form='log'):
    assert tensors0[0].shape[0] == tensors0[-1].shape[-1]
    assert tensors1[0].shape[0] == tensors1[-1].shape[-1]
    assert len(tensors0) == len(tensors1)

    v0 = tc.eye(tensors0[0].shape[0], dtype=tensors0[0].dtype, device=tensors0[0].device)
    v1 = tc.eye(tensors1[0].shape[0], dtype=tensors0[0].dtype, device=tensors0[0].device)
    v = tc.kron(v0, v1).reshape([tensors0[0].shape[0], tensors1[0].shape[0],
                                 tensors0[0].shape[0], tensors1[0].shape[0]])
    norm_list = list()
    for n in range(len(tensors0)):
        v = tc.einsum('uvap,adb,pdq->uvbq', v, tensors0[n].conj(), tensors1[n])
        norm_list.append(v.norm())
        v = v / norm_list[-1]
    if v.numel() > 1:
        norm1 = tc.einsum('acac->', v)
        norm_list.append(norm1)
    else:
        norm_list.append(v[0, 0, 0, 0])
    if form == 'log':  # 返回模方的log，舍弃符号
        norm = 0.0
        for x in norm_list:
            norm = norm + tc.log(x.abs())
    elif form == 'list':  # 返回列表
        return norm_list
    else:  # 直接返回模方
        norm = 1.0
        for x in norm_list:
            norm = norm * x
    return norm

def D_op(d, device):
    D_x = tc.zeros((d, d), dtype=tc.float64, device=device)
    for s in range(d-1):
        D_x[s, s+1] = mt.sqrt((s+1) / 2)
        D_x[s+1, s] = -mt.sqrt((s+1) / 2)
    return D_x

def X_op(d, device):
    X_x = tc.zeros((d, d), dtype=tc.float64, device=device)
    for s in range(d-1):
        X_x[s, s+1] = mt.sqrt((s+1) / 2)
        X_x[s+1, s] = mt.sqrt((s+1) / 2)
    return X_x

def op_mps(mps, D, X):
    mps_final = None
    d = mps[0].shape[1]

    #diffx**2
    # timex**2

    for bond in range(len(mps)):
        mps_ = clone_mps(mps)

        mps_[bond] = tc.einsum('abc,ib->aic',mps_[bond],D)
        mps_[0] = mps_[0] * -0.5
        if mps_final is None:
            mps_final = clone_mps(mps_)
        else:
            mps_final = add_two_mps(mps_final, mps_, bondary = 'par', auto_extend=False)

        mps_ = clone_mps(mps)

        mps_[bond] = tc.einsum('abc,ib->aic',mps_[bond],X)
        mps_[0] = mps_[0] * 0.5
        mps_final = add_two_mps(mps_final, mps_, bondary = 'par', auto_extend=False)

    #cp
    #
    '''
    for bond in range(paraa['length']-1):
        mps_ = clone_mps(mps)

        mps_[bond] = tc.einsum('abc,ib->aic',mps_[bond],times_x_ho(paraa['orderHermi']).to(choose_device()))
        mps_[bond+1] = tc.einsum('abc,ib->aic',mps_[bond+1],times_x_ho(paraa['orderHermi']).to(choose_device()))
        mps_[bond] = mps_[bond] * -0.5
        mps_final = add_two_mps(mps_final, mps_)
    '''
    return mps_final

