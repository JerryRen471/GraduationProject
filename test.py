import torch as tc
from Library import PhysModule as phy
from Library.TensorNetwork import TensorTrain, TensorNetwork
import time

device=tc.device('cpu')
dtype=tc.complex64
length = 5
a = TensorNetwork()
a.add_node(tc.rand(2, 2))
a.add_node(tc.rand(3, 3))
a.add_node(tc.rand(3, 4))
a.connect([1, 1], [2, 0])
a.merge_nodes((0, 1))
print(a.node_list[0].shape)
print(a.connect_graph)