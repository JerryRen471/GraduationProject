import torch as tc
import numpy as np
from Library.TensorNetwork import *

def J_SiSj(J, i, j, l, chi, dtype=tc.complex64, device=tc.device('cpu')):
    tn_x = TensorNetwork(chi=chi)
    tn_y = TensorNetwork(chi=chi)
    tn_z = TensorNetwork(chi=chi)
    Sx = tc.tensor([[0, 0.5], [0.5, 0]], dtype=dtype, device=device)
    Sy = tc.tensor([[0, -0.5j], [0.5j, 0]], dtype=dtype, device=device)
    Sz = tc.tensor([[0.5, 0], [0, -0.5]], dtype=dtype, device=device)
    for _ in range(i):
        tn_x.add_node(tc.eye(2, dtype=dtype, device=device))
        tn_y.add_node(tc.eye(2, dtype=dtype, device=device))
        tn_z.add_node(tc.eye(2, dtype=dtype, device=device))
    tn_x.add_node(Sx)
    tn_y.add_node(Sy)
    tn_z.add_node(Sz)
    for _ in range(i+1, j):
        tn_x.add_node(tc.eye(2, dtype=dtype, device=device))
        tn_y.add_node(tc.eye(2, dtype=dtype, device=device))
        tn_z.add_node(tc.eye(2, dtype=dtype, device=device))
    tn_x.add_node(Sx)
    tn_y.add_node(Sy)
    tn_z.add_node(Sz)
    for _ in range(j+1, l):
        tn_x.add_node(tc.eye(2, dtype=dtype, device=device))
        tn_y.add_node(tc.eye(2, dtype=dtype, device=device))
        tn_z.add_node(tc.eye(2, dtype=dtype, device=device))
    tn_x.merge_all()
    tn_y.merge_all()
    tn_z.merge_all()
    return tn_x.node_list[0]*J[0] + tn_y.node_list[0]*J[1] + tn_z.node_list[0]*J[2]

def h_Si(h, i, l, chi, dtype=tc.complex64, device=tc.device('cpu')):
    tn = TensorNetwork(chi)
    Sx = tc.tensor([[0, 0.5], [0.5, 0]], dtype=dtype, device=device)
    Sy = tc.tensor([[0, -0.5j], [0.5j, 0]], dtype=dtype, device=device)
    Sz = tc.tensor([[0.5, 0], [0, -0.5]], dtype=dtype, device=device)
    for _ in range(i):
        tn.add_node(tc.eye(2, dtype=dtype, device=device))
    tn.add_node(h[0]*Sx + h[1]*Sy + h[2]*Sz)
    for _ in range(i+1, l):
        tn.add_node(tc.eye(2, dtype=dtype, device=device))
    tn.merge_all()
    return tn.node_list[0]

def unfolding(spectrum):
    x = spectrum
    y = np.array([i for i in range(1, len(spectrum)+1)])
    pn = np.polyfit(x, y, deg=12)
    new_spectrum = np.polyval(pn, x)
    return new_spectrum

J = [1, 1, 1]
hl = 0
l = 10
chi = 5
import time
t1 = time.time()
H = 0
for i in range(l):
    hz = 0
    H -= h_Si([0, 0, hz], i, l, chi)
    if i < l-1:
        H += J_SiSj(J, i, i+1, l, chi)
    else:
        H += J_SiSj(J, 0, i, l, chi)
t2 = time.time()
print(t2-t1)
perm = [i*2 for i in range(l)] + [i*2+1 for i in range(l)]
H = tc.permute(H, perm).reshape([2**l, -1])

energy = tc.linalg.eigvals(H).real
energy = unfolding(energy)
print(energy)
t3 = time.time()
print(t3-t2)

import matplotlib.pyplot as plt

# 对张量进行排序
sorted_energy = np.sort(energy)
print(sorted_energy)

# 计算相邻元素之间的差值
differences = np.diff(sorted_energy)
print(differences.shape)

# 将张量转换为numpy数组以便于绘图
differences_np = differences
Sn = differences_np[:-1]
Snn = differences_np[1:]
# r = np.minimum(Sn, Snn)/np.maximum(Sn, Snn)


# 绘制差值的概率密度直方图并获取条形高度和区间边缘
counts, bin_edges = np.histogram(differences_np, bins=20, density=False)

# 绘制归一化能级间隔的直方图
plt.hist(differences_np, bins=50, density=True, label='normalized_spacings')

# 绘制泊松分布
s = np.linspace(0, 3, 100)
poisson_dist = np.exp(-s)
plt.plot(s, poisson_dist, 'r-', label='Poisson')

# 绘制GOE分布
goe_dist = (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)
plt.plot(s, goe_dist, 'g-', label='GOE')

plt.xlabel('s')
plt.ylabel('P(s)')
plt.legend()
plt.savefig('/data/home/scv7454/run/GraduationProject/pics/ProbDis.svg')
plt.close()

# plt.hist(r, bins=100, density=True, alpha=0.7, color='blue')
# plt.title('Probability Density of Differences Between Adjacent Elements')
# plt.xlabel('Difference')
# plt.ylabel('Probability Density')
# plt.grid(True)
# plt.savefig('/data/home/scv7454/run/GraduationProject/pics/ProbDis(r).svg')
# plt.show()
