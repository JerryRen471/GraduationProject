from Library.TensorNetwork import MPS_Dataset, TensorTrain_pack
from torch.utils.data import DataLoader

import torch as tc
from Library.TN_ADQC import *

length = 4
tensor_pack1 = [tc.rand(3,2, dtype=tc.complex64, requires_grad=False) for _ in range(length)]
tensor_pack2 = [tc.rand(3,2, dtype=tc.complex64, requires_grad=False) for _ in range(length)]
gate = tc.rand((4, 4), dtype=tc.complex64, requires_grad=True)
TTP1 = TensorTrain_pack(tensor_pack1, length=length)
dataset = MPS_Dataset(TTP1)

dataloader = DataLoader(dataset=dataset)
for data in dataloader:
    print(data)
    print()