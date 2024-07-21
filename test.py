import torch as tc
from Library import PhysModule as phy
from Library.TensorNetwork import TensorTrain, TensorNetwork
import time

device=tc.device('cpu')
dtype=tc.complex64
mps = tc.load('test.pt')
print(mps.center)
print(mps.chi)
print(len(mps.node_list))
print(mps.connect_graph)