from scipy import optimize
from Library.TensorNetwork import *
import torch as tc

def mps_multiply(mps1:TensorTrain, mps2:TensorTrain):
    phydim = mps1.phydim
    U = tc.rand(size=[phydim]*3)
    result_mps = TensorNetwork(chi=None, device=mps1.device, dtype=mps1.dtype)
    for i in range(mps1.length):
        l = len(result_mps.node_list)
        # 添加 mps1 的第 i 个局域张量，并和之前的局域张量进行连接
        result_mps.add_node(mps1.node_list[i])
        result_mps.connect([l - 1, 3], [l + 0, 0])
        # 添加转换矩阵 U
        result_mps.add_node(U)
        # 添加 mps2 的第 i 个局域张量，并和之前的局域张量进行连接
        result_mps.add_node(mps2.node_list[i])
        result_mps.connect([l - 1, 4], [l + 2, 0])

        # 连接转换矩阵与新添加的局域张量
        result_mps.connect([l + 0, 1], [l + 1, 0])
        result_mps.connect([l + 2, 1], [l + 1, 2])

        # 计算转换矩阵与局域张量的乘积，得到乘积 mps 的第 i 个局域张量
        result_mps.merge_nodes((0+l, 1+l))
        result_mps.merge_nodes((0+l, 1+l))
        # 对新得到的乘积 mps 的局域张量的角标进行重排，满足 ijk(mps1), jnp(U), lpq(mps2) -> ilnkq
        result_mps.permute_legs(0+l, [0, 3, 2, 1, 4])
    for i in range(len(result_mps) - 1):
        result_mps.flatten((i, i+1))
    return result_mps

params = [tc.rand(size=(2,4,2), dtype=tc.float64, requires_grad=True) for _ in range(9)]
optimizer = tc.optim.Adam(params, lr=0.01)

mps1 = [params[0], params[4], params[8]]
mps2 = [params[1], params[5], params[6]]
mps3 = [params[2], params[3], params[7]]

U = tc.rand((4,4,4), dtype=tc.float64)
H = tc.rand((4,4,4,4,4,4), dtype=tc.float64)

for i in range(10):
    print('第',i,'次迭代')
    K1 = tc.einsum('abc,ijk,bjt->aitck',mps1[0],mps2[0],U).reshape(4,4,4)
    K1 = tc.einsum('abc,ijk,bjt->aitck',K1,mps3[0],U).reshape(8,4,8)

    K2 = tc.einsum('abc,ijk,bjt->aitck',mps1[1],mps2[1],U).reshape(4,4,4)
    K2 = tc.einsum('abc,ijk,bjt->aitck',K2,mps3[1],U).reshape(8,4,8)

    K3 = tc.einsum('abc,ijk,bjt->aitck',mps1[2],mps2[2],U).reshape(4,4,4)
    K3 = tc.einsum('abc,ijk,bjt->aitck',K3,mps3[2],U).reshape(8,4,8)

    T = [K1,K2,K3]
    phi = tc.einsum('aib, bjc, cka -> ijk',T[0],T[1],T[2])
    loss = tc.einsum('abc, abcijk, ijk -> ', phi, H, phi)
    print(loss)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    print(params[0].grad)
    optimizer.step()