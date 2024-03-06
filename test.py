# from cv2 import mean
# import numpy as np
# import torch as tc

num_q = 10
X_even = list(range(0, num_q, 2))
X_odd = list(range(1, num_q-1, 2))
X_poses = [X_even, X_odd]
depth = 4
for nd in range(depth):
    for ng in X_poses[nd%2]:
        print('ng+1',ng+1)
        print('ng',ng)