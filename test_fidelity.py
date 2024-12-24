import torch

def matrix_sqrt(A):
    """
    计算复数矩阵 A 的平方根，使用特征值分解。
    
    参数：
    A (torch.Tensor): 复数矩阵。
    
    返回：
    torch.Tensor: 复数矩阵 A 的平方根。
    """
    # 进行特征值分解
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    
    # 计算特征值的平方根
    eigenvalues_sqrt = torch.sqrt(eigenvalues)
    
    # 构造平方根矩阵
    A_sqrt = eigenvectors @ torch.diag(eigenvalues_sqrt).to(dtype=torch.complex64) @ eigenvectors.T.conj()
    
    return A_sqrt

def fidelity(rho, sigma):
    """
    计算两个复数密度矩阵 rho 和 sigma 之间的保真度。
    
    参数：
    rho (torch.Tensor): 复数密度矩阵 rho。
    sigma (torch.Tensor): 复数密度矩阵 sigma。
    
    返回：
    float: 两个密度矩阵之间的保真度。
    """
    # 计算 sqrt(rho)
    sqrt_rho = matrix_sqrt(rho)  # 使用自定义的 matrix_sqrt 函数计算复数矩阵的平方根
    
    # 计算 sqrt(rho) * sigma * sqrt(rho)
    product = torch.matmul(torch.matmul(sqrt_rho, sigma), sqrt_rho)
    
    # 计算 sqrt(sqrt(rho) * sigma * sqrt(rho))
    sqrt_product = matrix_sqrt(product)  # 计算 sqrt(sqrt(rho) * sigma * sqrt(rho))
    
    # 计算保真度 F = Tr(sqrt(sqrt(rho) * sigma * sqrt(rho)))
    return torch.abs(torch.trace(sqrt_product))**2

# 示例：两个复数密度矩阵
state1 = torch.tensor([1, 0], dtype=torch.complex64)
state2 = torch.tensor([1, 1.j], dtype=torch.complex64) / 2**0.5
rho = torch.einsum('i,j->ij', state1, state1.conj())
sigma = torch.einsum('i,j->ij', state2, state2.conj()) * 0.5 + torch.einsum('i,j->ij', state1, state1.conj()) * 0.5
# rho = torch.tensor([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=torch.complex64)  # 复数矩阵 rho
# sigma = torch.tensor([[0.5 - 0.5j, -0.5 + 0.5j], [-0.5 + 0.5j, 0.5 + 0.5j]], dtype=torch.complex64)  # 复数矩阵 sigma

# 计算并打印保真度
print(fidelity(sigma, sigma))
