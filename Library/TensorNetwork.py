from copy import deepcopy
from platform import node
import time
from turtle import left, right
# from debugpy import connect
import torch as tc
import numpy as np
from Library.BasicFun import pad_and_cat
from rand_uni import n_body_evol_states
# from traitlets import default

def copy_from_tn(tn):
    new_tn = TensorNetwork(tn.chi, device=tn.device, dtype=tn.dtype)
    new_tn.node_list = tn.node_list[:]
    copy = lambda x: x[:]
    new_tn.connect_graph = dict(zip(tn.connect_graph.keys(), map(copy, tn.connect_graph.values())))
    new_tn.device = tn.device
    new_tn.dtype = tn.dtype
    new_tn.history = tn.history
    return new_tn

class TensorNetwork():
    def __init__(self, chi=None, device=tc.device('cpu'), dtype=tc.complex64) -> None:
        self.node_list:list = []
        self.chi = chi
        self.connect_graph = dict()
        self.device = device
        self.dtype = dtype
        self.history = {'split_node':[], 'merge_nodes':[], 'move_node':[]}
        self.trunc_error = 0
    
    def copy_from_tn(self, tn):
        self.node_list = tn.node_list[:]
        self.chi = tn.chi
        copy = lambda x: x[:]
        self.connect_graph = dict(zip(self.connect_graph.keys(), map(copy, self.connect_graph.values())))
        self.device = tn.device
        self.dtype = tn.dtype
        self.history = tn.history

    def add_node(self, node:tc.Tensor, site=-1, device=tc.device('cpu'), dtype=tc.complex64):
        self.node_list.append(node)
        if site != -1:
            self.move_node(-1, site)

    def connect(self, node_leg1:list, node_leg2:list):
        """
        输入的 node_leg 格式将被标准化：
        1.没有负数; 
        2.node1 < node2
        """
        # 将输入的 node_leg 格式标准化：1.没有负数; 2.node1<node2
        if node_leg1[0] < 0:
            node_leg1[0] = len(self.node_list)+node_leg1[0]
        if node_leg2[0] < 0:
            node_leg2[0] = len(self.node_list)+node_leg2[0]
        node_leg1[1] = node_leg1[1] % self.node_list[node_leg1[0]].dim()
        node_leg2[1] = node_leg2[1] % self.node_list[node_leg2[0]].dim()
        if node_leg1[0] > node_leg2[0]:
            tmp = node_leg1
            node_leg1 = node_leg2
            node_leg2 = tmp
        
        node1 = node_leg1[0]
        leg1 = node_leg1[1]
        node2 = node_leg2[0]
        leg2 = node_leg2[1]
        node_pair = (node1, node2)
        leg_pair = (leg1, leg2)
        connected_legs = []
        if node_pair in self.connect_graph.keys():
            connected_legs = self.connect_graph[node_pair]
        if leg_pair not in connected_legs:
            connected_legs.append(leg_pair)
        self.connect_graph[node_pair] = connected_legs
        pass

    def renew_graph(self):
        splits = self.history['split_node']
        merges = self.history['merge_nodes']
        moves = self.history['move_node']
        self.split_graph(splits)
        self.merge_graph(merges)
        self.move_graph(moves)
        pass

    def split_graph(self, splits):
        for i in range(len(splits)):
            node_idx, legs_L, legs_R = splits.pop()
            keys = list(self.connect_graph.keys())[:]
            tmp_dict = dict()
            for node_pair in keys:
                nodes = list(node_pair)
                legs_old_list = self.connect_graph.pop(node_pair)
                legs_pair_list_new = []
                for legs_old in legs_old_list:
                    legs_pair_new = list(legs_old) # 新建一个保存新的腿连接关系的列表
                    # 修正node_pair的第一个node
                    # 如果是split的node，则对这个node对应的leg进行修正
                    for j in range(2):
                        if node_pair[j] == node_idx:
                            # 对node之间所有连接的腿进行修正
                            # 连接的腿属于legs_L
                            if legs_old[j] in legs_L:
                                legs_pair_new[j] = legs_L.index(legs_old[j]) # 用以前的leg在legs_L中的位置作为新的位置
                                nodes[j] = node_idx
                            # 连接的腿属于legs_R
                            elif legs_old[j] in legs_R:
                                legs_pair_new[j] = legs_R.index(legs_old[j])+1
                                nodes[j] = node_idx+2
                        # 如果是排在split的node之后的node，则对应的node序号加二
                        elif node_pair[j] > node_idx:
                            nodes[j] = node_pair[j]+2
                    legs_pair_list_new.append(tuple(legs_pair_new))
                tmp_dict[tuple(sorted(nodes))] = legs_pair_list_new
            self.connect_graph.update(tmp_dict)
            self.connect_graph[(node_idx, node_idx+1)] = [(len(legs_L), 0)]
            self.connect_graph[(node_idx+1, node_idx+2)] = [(1, 0)]

    def merge_graph(self, merges):
        for i in range(len(merges)):
            node_pair, n1_left_legs, n2_left_legs = merges.pop()
            if node_pair in self.connect_graph.keys():
                self.connect_graph.pop(node_pair)
            keys = list(self.connect_graph.keys())[:]
            dict_tmp = dict()
            for key in keys:
                new_leg_pairs = [] # ?
                old_leg_pairs = self.connect_graph.pop(key)
                for old_leg_pair in old_leg_pairs:
                    new_node_pair = [None, None]
                    new_leg_pair = list(old_leg_pair)
                    for j in range(2):
                        if key[j] == node_pair[0]:
                            new_node_pair[j] = node_pair[0]
                            new_leg_pair[j] = n1_left_legs.index(old_leg_pair[j])
                            pass
                        elif key[j] == node_pair[1]:
                            new_node_pair[j] = node_pair[0]
                            new_leg_pair[j] = len(n1_left_legs)+n2_left_legs.index(old_leg_pair[j])
                            pass
                        elif key[j] > node_pair[1]:
                            new_node_pair[j] = key[j]-1
                        else:
                            new_node_pair[j] = key[j]
                    if new_node_pair[0] > new_node_pair[1]:
                        tmp = new_node_pair[0]
                        new_node_pair[0] = new_node_pair[1]
                        new_node_pair[1] = tmp
                        tmp = new_leg_pair[0]
                        new_leg_pair[0] = new_leg_pair[1]
                        new_leg_pair[1] = tmp
                    new_leg_pairs.append(tuple(new_leg_pair))
                new_leg_pairs = dict_tmp.get(tuple(new_node_pair), []) + new_leg_pairs
                dict_tmp[tuple(new_node_pair)] = new_leg_pairs
            self.connect_graph.update(dict_tmp)

    def move_graph(self, moves):
        idx = list(i for i in range(len(self.node_list)))
        for i in range(len(moves)):
            move = moves.pop(0)
            move_from = move[0]
            move_to = move[1]
            if move_from < move_to:
                idx = idx[:move_from] + idx[move_from+1:move_to+1] + idx[move_from:move_from+1] + idx[move_to+1:]
            elif move_from > move_to:
                idx = idx[:move_to] + idx[move_from:move_from+1] + idx[move_to:move_from] + idx[move_from+1:]
        keys = list(self.connect_graph.keys())[:]
        tmp_dict = dict()
        for node_pair in keys:
            nodes = [idx.index(node_pair[0]), idx.index(node_pair[1])]
            legs_list = self.connect_graph.pop(node_pair)
            if nodes[0] > nodes[1]:
                legs_list = [(legs[1], legs[0]) for legs in legs_list]
                pass
            tmp_dict[tuple(sorted(nodes))] = legs_list
        self.connect_graph.update(tmp_dict)

    def move_node(self, node_from, node_to):
        node_from = node_from % len(self.node_list)
        node_to = node_to % len(self.node_list)
        tmp = self.node_list[:]
        if node_from < node_to:
            self.node_list = tmp[:node_from] + tmp[node_from+1:node_to+1] + tmp[node_from:node_from+1] + tmp[node_to+1:]
        elif node_from > node_to:
            self.node_list = tmp[:node_to] + tmp[node_from:node_from+1] + tmp[node_to:node_from] + tmp[node_from+1:]
        self.history['move_node'].append((node_from, node_to))
        self.renew_graph()
        pass

    def get_merged_node(self, node_pair:tuple):
        node_pair = list(node_pair)
        for i in range(2):
            if node_pair[i] < 0:
                node_pair[i] = len(self.node_list) + node_pair[i]
        node_pair = tuple(sorted(node_pair))
        node1 = self.node_list[node_pair[0]]
        node2 = self.node_list[node_pair[1]]
        n1_left_legs = [_ for _ in range(len(node1.shape))]
        n2_left_legs = [_ for _ in range(len(node2.shape))]
        leg_pairs = self.connect_graph.get(node_pair, [])
        n1_merge_legs = []
        n2_merge_legs = []
        for leg_piar in leg_pairs:
            n1_merge_legs.append(leg_piar[0])
            n2_merge_legs.append(leg_piar[1])
            n1_left_legs.remove(leg_piar[0])
            n2_left_legs.remove(leg_piar[1])
        new_node = tc.tensordot(node1, node2, dims=(n1_merge_legs, n2_merge_legs))
        return new_node, n1_left_legs, n2_left_legs

    def merge_nodes(self, node_pair:tuple):
        node_pair = list(node_pair)
        for i in range(2):
            if node_pair[i] < 0:
                node_pair[i] = len(self.node_list) + node_pair[i]
        node_pair = tuple(sorted(node_pair))
        new_node, n1_left_legs, n2_left_legs = self.get_merged_node(node_pair)
        self.node_list = self.node_list[:node_pair[0]] + [new_node] + self.node_list[node_pair[0]+1:node_pair[1]] + self.node_list[node_pair[1]+1:]
        self.history['merge_nodes'].append([node_pair, n1_left_legs, n2_left_legs])
        self.renew_graph()

    def permute_legs(self, node_idx:int, perm:list):
        node = self.node_list[node_idx]
        self.node_list[node_idx] = tc.permute(node, perm)
        keys = list(self.connect_graph.keys())[:]
        for key in keys:
            for i in range(len(key)):
                if key[i] == node_idx:
                    connect_legs = self.connect_graph[key]
                    for j, legs in enumerate(connect_legs):
                        tmp = list(legs)
                        tmp[i] = perm[legs[i]]
                        connect_legs[j] = tuple(tmp)
                    self.connect_graph[key] = connect_legs
        pass

    def merge_all(self):
        for i in range(len(self.node_list)-1):
            self.merge_nodes((0, 1))

    def split_node(self, node_idx:int, legs_group:list, group_side:str='L', if_trun=True):
        if node_idx < 0:
            node_idx = node_idx + len(self.node_list)
        node = self.node_list[node_idx]
        dims = node.shape
        if group_side == 'L':
            legs_L = [_ % node.dim() for _ in legs_group]
            legs_R = [_ for _ in range(node.dim())]
            for i in legs_L:
                legs_R.remove(i)
        elif group_side == 'R':
            legs_R = [_ % node.dim() for _ in legs_group]
            legs_L = [_ for _ in range(node.dim())]
            for i in legs_R:
                legs_L.remove(i)
        else:
            print("!WARNING! para: 'group_side' must be either 'L' or 'R'")
        dimL = 1
        for i in legs_L:
            dimL = dimL * dims[i]
        perm = legs_L[:] + legs_R[:]
        node = node.permute(perm)
        u, s, v = tc.linalg.svd(node.reshape(dimL, -1), full_matrices=False)
        if if_trun:
            if self.chi == None:
                dc = s.numel()
            else:
                dc = min(self.chi, s.numel())
            trunc_error = tc.sum(s[dc:])
        else:
            dc = s.numel()
            trunc_error = 0
        self.trunc_error += trunc_error
        u = u[:, :dc].reshape([dims[i] for i in legs_L]+[dc])
        s = tc.diag(s[:dc])
        v = v[:dc, :].reshape([dc]+[dims[i] for i in legs_R])
        self.node_list = self.node_list[:node_idx] + [u, \
                                                        s, \
                                                        v] + self.node_list[node_idx+1:]
        self.history['split_node'].append([node_idx, legs_L[:], legs_R[:]])
        self.renew_graph()
        pass

    def flatten(self, node_pair:tuple):
        nodes = list(node_pair)
        node1 = self.node_list[nodes[0]]
        node2 = self.node_list[nodes[1]]
        left1 = list(i for i in range(node1.dim()))
        left2 = list(i for i in range(node2.dim()))
        legs_old_list = self.connect_graph.pop(node_pair)
        legs1 = []
        legs2 = []
        shape1 = []
        shape2 = []
        dim = 1
        for leg_pair in legs_old_list:
            legs1.append(leg_pair[0])
            legs2.append(leg_pair[1])
            left1.remove(leg_pair[0])
            left2.remove(leg_pair[1])
            dim = dim * node1.shape[leg_pair[0]]
        for i in left1:
            shape1.append(node1.shape[i])
        for j in left2:
            shape2.append(node2.shape[j])
        shape1 = shape1[:] + [dim]
        shape2 = [dim] + shape2[:]
        perm1 = left1[:] + legs1[:]
        perm2 = legs2[:] + left2[:]
        node1 = tc.permute(node1, perm1).reshape(shape1)
        node2 = tc.permute(node2, perm2).reshape(shape2)
        self.node_list[nodes[0]] = node1
        self.node_list[nodes[1]] = node2
        self.connect_graph[node_pair] = [(node1.dim()-1, 0)]
        # 处理connect_graph
        left1 = left1 + [None]
        left2 = [None] + left2
        keys = list(self.connect_graph.keys())[:]
        keys.remove(node_pair)
        for key in keys:
            old_legs = self.connect_graph[key]
            for j in range(2):
                if key[j] == node_pair[0]:
                    tmp = old_legs[:]
                    for i, leg_pair in enumerate(tmp):
                        leg_pair = list(leg_pair)
                        leg_pair[j] = left1.index(leg_pair[j])
                        old_legs[i] = tuple(leg_pair)
                elif key[j] == node_pair[1]:
                    tmp = old_legs[:]
                    for i, leg_pair in enumerate(tmp):
                        leg_pair = list(leg_pair)
                        leg_pair[j] = left2.index(leg_pair[j])
                        old_legs[i] = tuple(leg_pair)
            self.connect_graph[key] = old_legs

class TensorNetwork_pack():
    def __init__(self, chi=None, device=tc.device('cpu'), dtype=tc.complex64) -> None:
        self.node_list:list = []
        self.chi = chi
        self.connect_graph = dict()
        self.device = device
        self.dtype = dtype
        self.history = {'split_node':[], 'merge_nodes':[], 'move_node':[]}
        self.trunc_error = tc.tensor(0, dtype=dtype, device=device)
    
    def copy_from_tn(self, tn):
        self.node_list = tn.node_list[:]
        self.chi = tn.chi
        copy = lambda x: x[:]
        self.connect_graph = dict(zip(self.connect_graph.keys(), map(copy, self.connect_graph.values())))
        self.device = tn.device
        self.dtype = tn.dtype
        self.history = tn.history

    def add_node(self, node:tc.Tensor, site=-1, device=tc.device('cpu'), dtype=tc.complex64):
        self.node_list.append(node.to(device=device, dtype=dtype))
        if site != -1:
            self.move_node(-1, site)

    def connect(self, node_leg1:list, node_leg2:list):
        """
        输入的 node_leg 格式将被标准化：
        1.没有负数; 
        2.node1 < node2
        """
        # 将输入的 node_leg 格式标准化：1.没有负数; 2.node1<node2
        if node_leg1[0] < 0:
            node_leg1[0] = len(self.node_list)+node_leg1[0]
        if node_leg2[0] < 0:
            node_leg2[0] = len(self.node_list)+node_leg2[0]
        node_leg1[1] = node_leg1[1] % self.node_list[node_leg1[0]].dim()
        node_leg2[1] = node_leg2[1] % self.node_list[node_leg2[0]].dim()
        if node_leg1[0] > node_leg2[0]:
            tmp = node_leg1
            node_leg1 = node_leg2
            node_leg2 = tmp
        
        node1 = node_leg1[0]
        leg1 = node_leg1[1]
        node2 = node_leg2[0]
        leg2 = node_leg2[1]
        node_pair = (node1, node2)
        leg_pair = (leg1, leg2)
        connected_legs = []
        if node_pair in self.connect_graph.keys():
            connected_legs = self.connect_graph[node_pair]
        if leg_pair not in connected_legs:
            connected_legs.append(leg_pair)
        self.connect_graph[node_pair] = connected_legs
        pass

    def renew_graph(self):
        splits = self.history['split_node']
        merges = self.history['merge_nodes']
        moves = self.history['move_node']
        self.split_graph(splits)
        self.merge_graph(merges)
        self.move_graph(moves)
        pass

    def split_graph(self, splits):
        for i in range(len(splits)):
            node_idx, legs_L, legs_R = splits.pop()
            keys = list(self.connect_graph.keys())[:]
            tmp_dict = dict()
            for node_pair in keys:
                nodes = list(node_pair)
                legs_old_list = self.connect_graph.pop(node_pair)
                legs_pair_list_new = []
                for legs_old in legs_old_list:
                    legs_pair_new = list(legs_old) # 新建一个保存新的腿连接关系的列表
                    # 修正node_pair的第一个node
                    # 如果是split的node，则对这个node对应的leg进行修正
                    for j in range(2):
                        if node_pair[j] == node_idx:
                            # 对node之间所有连接的腿进行修正
                            # 连接的腿属于legs_L
                            if legs_old[j] in legs_L:
                                legs_pair_new[j] = legs_L.index(legs_old[j])+1 # 用以前的leg在legs_L中的位置作为新的位置
                                nodes[j] = node_idx
                            # 连接的腿属于legs_R
                            elif legs_old[j] in legs_R:
                                legs_pair_new[j] = legs_R.index(legs_old[j])+1+1
                                nodes[j] = node_idx+2
                        # 如果是排在split的node之后的node，则对应的node序号加二
                        elif node_pair[j] > node_idx:
                            nodes[j] = node_pair[j]+2
                    legs_pair_list_new.append(tuple(legs_pair_new))
                tmp_dict[tuple(sorted(nodes))] = legs_pair_list_new
            self.connect_graph.update(tmp_dict)
            self.connect_graph[(node_idx, node_idx+1)] = [(len(legs_L)+1, 1)]
            self.connect_graph[(node_idx+1, node_idx+2)] = [(2, 1)]

    def merge_graph(self, merges):
        def renew_leg_pairs(renew_node, old_leg_pairs, leg_map):
            new_leg_pairs = []
            for old_leg_pair in old_leg_pairs:
                new_leg_pair = list(old_leg_pair)
                new_leg_pair[renew_node] = leg_map[old_leg_pair[renew_node]]
                new_leg_pairs.append(tuple(new_leg_pair))
            return new_leg_pairs

        for i in range(len(merges)):
            node_pair, n1_map, n2_map = merges.pop()
            if node_pair in self.connect_graph.keys():
                self.connect_graph.pop(node_pair)
            keys = list(self.connect_graph.keys())[:]
            dict_tmp = dict()
            for key in keys:
                new_leg_pairs = []
                old_leg_pairs = self.connect_graph.pop(key)
                if key[0] < node_pair[0]:
                    if key[1] < node_pair[0]:
                        new_node_pair = [key[0], key[1]]
                        new_leg_pairs = deepcopy(old_leg_pairs)
                    elif key[1] == node_pair[0]:
                        new_node_pair = [key[0], key[1]]
                        new_leg_pairs = renew_leg_pairs(renew_node=1, old_leg_pairs=old_leg_pairs, leg_map=n1_map)
                        pass
                    elif key[1] > node_pair[0] and key[1] < node_pair[1]:
                        new_node_pair = [key[0], key[1]]
                        new_leg_pairs = deepcopy(old_leg_pairs)
                        pass
                    elif key[1] == node_pair[1]:
                        new_node_pair = [key[0], node_pair[0]]
                        new_leg_pairs = renew_leg_pairs(renew_node=1, old_leg_pairs=old_leg_pairs, leg_map=n2_map)
                        pass
                    elif key[1] > node_pair[1]:
                        new_node_pair = [key[0], key[1]-1]
                        new_leg_pairs = deepcopy(old_leg_pairs)
                        pass
                    pass
                elif key[0] == node_pair[0]:
                    if key[1] < node_pair[1]:
                        new_node_pair = [node_pair[0], key[1]]
                        new_leg_pairs = renew_leg_pairs(renew_node=0, old_leg_pairs=old_leg_pairs, leg_map=n1_map)
                        pass
                    elif key[1] == node_pair[1]:
                        pass
                    elif key[1] > node_pair[1]:
                        new_node_pair = [node_pair[0], key[1]-1]
                        new_leg_pairs = renew_leg_pairs(renew_node=0, old_leg_pairs=old_leg_pairs, leg_map=n1_map)
                        pass
                    pass
                elif key[0] > node_pair[0] and key[0] < node_pair[1]:
                    if key[1] < node_pair[1]:
                        new_node_pair = [key[0], key[1]]
                        new_leg_pairs = deepcopy(old_leg_pairs)
                        pass
                    elif key[1] == node_pair[1]:
                        new_node_pair = [node_pair[0], key[0]]
                        tmp_leg_pairs = renew_leg_pairs(renew_node=1, old_leg_pairs=old_leg_pairs, leg_map=n2_map)
                        new_leg_pairs = []
                        for leg_pair in tmp_leg_pairs:
                            new_leg_pairs.append(tuple(leg_pair[1], leg_pair[0]))
                        pass
                    elif key[1] > node_pair[1]:
                        new_node_pair = [key[0], key[1]-1]
                        new_leg_pairs = deepcopy(old_leg_pairs)
                        pass
                    pass
                elif key[0] == node_pair[1]:
                    if key[1] > node_pair[1]:
                        new_node_pair = [node_pair[0], key[1]-1]
                        new_leg_pairs = renew_leg_pairs(renew_node=0, old_leg_pairs=old_leg_pairs, leg_map=n2_map)
                        pass
                    pass
                elif key[0] > node_pair[1]:
                    new_node_pair = [key[0]-1, key[1]-1]
                    new_leg_pairs = deepcopy(old_leg_pairs)
                    pass
                new_leg_pairs = dict_tmp.get(tuple(new_node_pair), []) + new_leg_pairs
                dict_tmp[tuple(new_node_pair)] = new_leg_pairs
            self.connect_graph.update(dict_tmp)

    '''
    def merge_graph(self, merges):
        for i in range(len(merges)):
            node_pair, n1_left_legs, n2_left_legs = merges.pop()
            if node_pair in self.connect_graph.keys():
                self.connect_graph.pop(node_pair)
            keys = list(self.connect_graph.keys())[:]
            dict_tmp = dict()
            for key in keys:
                new_leg_pairs = []
                old_leg_pairs = self.connect_graph.pop(key)
                for old_leg_pair in old_leg_pairs:
                    new_node_pair = [None, None]
                    new_leg_pair = list(old_leg_pair)
                    for j in range(2):
                        if key[j] == node_pair[0]:
                            new_node_pair[j] = node_pair[0]
                            new_leg_pair[j] = n1_left_legs.index(old_leg_pair[j])
                            pass
                        elif key[j] == node_pair[1]:
                            new_node_pair[j] = node_pair[0]
                            new_leg_pair[j] = len(n1_left_legs)+n2_left_legs.index(old_leg_pair[j])
                            pass
                        elif key[j] > node_pair[1]:
                            new_node_pair[j] = key[j]-1
                        else:
                            new_node_pair[j] = key[j]
                    if new_node_pair[0] > new_node_pair[1]:
                        tmp = new_node_pair[0]
                        new_node_pair[0] = new_node_pair[1]
                        new_node_pair[1] = tmp
                        tmp = new_leg_pair[0]
                        new_leg_pair[0] = new_leg_pair[1]
                        new_leg_pair[1] = tmp
                    new_leg_pairs.append(tuple(new_leg_pair))
                new_leg_pairs = dict_tmp.get(tuple(new_node_pair), []) + new_leg_pairs
                dict_tmp[tuple(new_node_pair)] = new_leg_pairs
            self.connect_graph.update(dict_tmp)
    '''

    def move_graph(self, moves):
        idx = list(i for i in range(len(self.node_list)))
        for i in range(len(moves)):
            move = moves.pop(0)
            move_from = move[0]
            move_to = move[1]
            if move_from < move_to:
                idx = idx[:move_from] + idx[move_from+1:move_to+1] + idx[move_from:move_from+1] + idx[move_to+1:]
            elif move_from > move_to:
                idx = idx[:move_to] + idx[move_from:move_from+1] + idx[move_to:move_from] + idx[move_from+1:]
        keys = list(self.connect_graph.keys())[:]
        tmp_dict = dict()
        for node_pair in keys:
            nodes = [idx.index(node_pair[0]), idx.index(node_pair[1])]
            legs_list = self.connect_graph.pop(node_pair)
            if nodes[0] > nodes[1]:
                legs_list = [(legs[1], legs[0]) for legs in legs_list]
                pass
            tmp_dict[tuple(sorted(nodes))] = legs_list
        self.connect_graph.update(tmp_dict)

    def move_node(self, node_from, node_to):
        node_from = node_from % len(self.node_list)
        node_to = node_to % len(self.node_list)
        tmp = self.node_list[:]
        if node_from < node_to:
            self.node_list = tmp[:node_from] + tmp[node_from+1:node_to+1] + tmp[node_from:node_from+1] + tmp[node_to+1:]
        elif node_from > node_to:
            self.node_list = tmp[:node_to] + tmp[node_from:node_from+1] + tmp[node_to:node_from] + tmp[node_from+1:]
        self.history['move_node'].append((node_from, node_to))
        self.renew_graph()
        pass

    def get_merged_node(self, node_pair:tuple, is_gate:tuple=(False, False)):
        node_pair = list(node_pair)
        for i in range(2):
            if node_pair[i] < 0:
                node_pair[i] = len(self.node_list) + node_pair[i]
        node_pair = tuple(sorted(node_pair))
        node1 = self.node_list[node_pair[0]]
        node2 = self.node_list[node_pair[1]]
        start_leg = lambda x: 0 if x==True else 1
        n1_left_legs = [_ for _ in range(start_leg(is_gate[0]), len(node1.shape))]
        n2_left_legs = [_ for _ in range(start_leg(is_gate[1]), len(node2.shape))]
        
        # if is_gate[0] == True and is_gate[1] == True:
        #     n1_left_legs = [_ for _ in range(0, len(node1.shape))]
        #     n2_left_legs = [_ for _ in range(0, len(node2.shape))]
        # elif is_gate[0] == True and is_gate[1] == False:
        #     n1_left_legs = [_ for _ in range(0, len(node1.shape))]
        #     n2_left_legs = [_ for _ in range(0, len(node2.shape))]
        # elif is_gate[0] == False and is_gate[1] == True:
        #     n1_left_legs = [_ for _ in range(0, len(node1.shape))]
        #     n2_left_legs = [_ for _ in range(0, len(node2.shape))]
        # elif is_gate[0] == False and is_gate[1] == False:
        #     n1_left_legs = [_ for _ in range(0, len(node1.shape))]
        #     n2_left_legs = [_ for _ in range(1, len(node2.shape))]
        leg_pairs = self.connect_graph.get(node_pair, [])
        n1_merge_legs = []
        n2_merge_legs = []
        dim1 = 1
        dim2 = 1
        for leg_piar in leg_pairs:
            n1_merge_legs.append(leg_piar[0])
            dim1 = dim1 * node1.shape[leg_piar[0]]
            n2_merge_legs.append(leg_piar[1])
            dim2 = dim2 * node2.shape[leg_piar[1]]
            n1_left_legs.remove(leg_piar[0])
            n2_left_legs.remove(leg_piar[1])

        if is_gate[0] == True and is_gate[1] == True:
            perm_1 = n1_left_legs+n1_merge_legs
            perm_2 = n2_merge_legs+n2_left_legs
            shape_1 = [-1, dim1]
            shape_2 = [dim2, -1]
            einsum_str = 'ij,jk->ik'
            new_shape = [node1.shape[i] for i in n1_left_legs]+[node2.shape[j] for j in n2_left_legs]
            n1_new_legs = list(i for i in range(len(n1_left_legs)))
            n2_new_legs = list(i + len(n1_left_legs) for i in range(len(n2_left_legs)))
        elif is_gate[0] == True and is_gate[1] == False:
            perm_1 = n1_left_legs+n1_merge_legs
            perm_2 = [0]+n2_merge_legs+n2_left_legs
            shape_1 = [-1, dim1]
            shape_2 = [node2.shape[0], dim2, -1]
            einsum_str = 'ij,njk->nik'
            new_shape = [node2.shape[0]]+[node1.shape[i] for i in n1_left_legs]+[node2.shape[j] for j in n2_left_legs]
            n1_new_legs = list(i + 1 for i in range(len(n1_left_legs)))
            n2_new_legs = list(i + 1 + len(n1_left_legs) for i in range(len(n2_left_legs)))
        elif is_gate[0] == False and is_gate[1] == True:
            perm_1 = [0]+n1_left_legs+n1_merge_legs
            perm_2 = n2_merge_legs+n2_left_legs
            shape_1 = [node1.shape[0], -1, dim1]
            shape_2 = [dim2, -1]
            einsum_str = 'nij,jk->nik'
            new_shape = [node1.shape[0]]+[node1.shape[i] for i in n1_left_legs]+[node2.shape[j] for j in n2_left_legs]
            n1_new_legs = list(i + 1 for i in range(len(n1_left_legs)))
            n2_new_legs = list(i + len(n1_left_legs) + 1 for i in range(len(n2_left_legs)))
        elif is_gate[0] == False and is_gate[1] == False:
            perm_1 = [0]+n1_left_legs+n1_merge_legs
            perm_2 = [0]+n2_merge_legs+n2_left_legs
            shape_1 = [node1.shape[0], -1, dim1]
            shape_2 = [node2.shape[0], dim2, -1]
            einsum_str = 'nij,njk->nik'
            new_shape = [node1.shape[0]]+[node1.shape[i] for i in n1_left_legs]+[node2.shape[j] for j in n2_left_legs]
            n1_new_legs = list(i + 1 for i in range(len(n1_left_legs)))
            n2_new_legs = list(i + 1 + len(n1_left_legs) for i in range(len(n2_left_legs)))

        try:
            node1 = node1.permute(perm_1).reshape(shape_1)
            node2 = node2.permute(perm_2).reshape(shape_2)
            new_node = tc.einsum(einsum_str, node1, node2).reshape(new_shape)
        except Exception as e:
            # 监测node1_和node2_的显存占用
            def print_mem(tensor, name):
                mem_bytes = tensor.element_size() * tensor.nelement()
                mem_mb = mem_bytes / 1024 / 1024
                device = "GPU" if tensor.is_cuda else "CPU"
                print(f"[TensorNetwork] {name} {device} memory usage: {mem_mb:.2f} MB, shape: {tensor.shape}")
            print("[TensorNetwork] RuntimeError in einsum/reshape:", str(e))
            print("[TensorNetwork] chi is", self.chi)
            print_mem(node1, "node1")
            print_mem(node2, "node2")
            # 预估new_node的shape和显存
            est_elem = 1
            for s in new_shape:
                est_elem *= s
            est_bytes = est_elem * node1.element_size()
            est_mb = est_bytes / 1024 / 1024
            print(f"[TensorNetwork] Estimated new_node memory usage: {est_mb:.2f} MB, shape: {new_shape}")
            raise
        n1_map = dict(zip(n1_left_legs, n1_new_legs))
        n2_map = dict(zip(n2_left_legs, n2_new_legs))
        return new_node, n1_map, n2_map

    def merge_nodes(self, node_pair:tuple, is_gate:tuple=(False, False)):
        node_pair = list(node_pair)
        for i in range(2):
            if node_pair[i] < 0:
                node_pair[i] = len(self.node_list) + node_pair[i]
        node_pair = tuple(sorted(node_pair))
        new_node, n1_map, n2_map = self.get_merged_node(node_pair, is_gate=is_gate)
        self.node_list = self.node_list[:node_pair[0]] + [new_node] + self.node_list[node_pair[0]+1:node_pair[1]] + self.node_list[node_pair[1]+1:]
        self.history['merge_nodes'].append([node_pair, n1_map, n2_map])
        self.renew_graph()

    def permute_legs(self, node_idx:int, cycle:list):
        '''
        perm: 局域张量指标交换后的顺序，注意从1开始
        '''
        def convert_cycle_to_full_sequence(cycle, length):
            # 初始化完整序列为原始顺序
            full_sequence = list(range(length))
            
            # 应用轮换表示
            for i in range(len(cycle) - 1):
                full_sequence[cycle[i]], full_sequence[cycle[i + 1]] = full_sequence[cycle[i + 1]], full_sequence[cycle[i]]
            
            return full_sequence
        perm = convert_cycle_to_full_sequence(cycle, self.node_list[node_idx].dim())
        node = self.node_list[node_idx]
        self.node_list[node_idx] = tc.permute(node, perm)
        keys = list(self.connect_graph.keys())[:]
        for key in keys:
            for i in range(len(key)):
                if key[i] == node_idx:
                    connect_legs = self.connect_graph[key]
                    for j, legs in enumerate(connect_legs):
                        tmp = list(legs)
                        tmp[i] = perm.index(legs[i])
                        connect_legs[j] = tuple(tmp)
                    self.connect_graph[key] = connect_legs
        pass

    def merge_all(self):
        for i in range(len(self.node_list)-1):
            self.merge_nodes((0, 1))

    @staticmethod
    def find_first_column_with_element_magnitude_less_than_e(tensor, e):
        n, d = tensor.shape

        left, right = 0, d - 1
        
        while left <= right:
            mid = (left + right) // 2
            # 检查中间列是否有模长小于 e 的元素
            if tc.any(tc.abs(tensor[:, mid]) < e):
                # 如果中间列有模长小于 e 的元素，则继续检查左半部分
                right = mid - 1
            else:
                # 如果中间列没有模长小于 e 的元素，则继续检查右半部分
                left = mid + 1

        # 检查找到的左边界列
        if left < d and tc.any(tc.abs(tensor[:, left]) < e):
            return left

        return d  # 如果没有找到这样的列，返回 tensor 的总列数

    def split_node(self, node_idx:int, legs_group:list, group_side:str='L', if_trun=True):
        '''
        将局域张量按照 legs_group 分组，group_side 指定分组给出的指标在左边还是右边，剩下的指标分给对应的一边。如 group_side 为 'L' ，则将 legs_group 分在左边，局域张量剩下的指标分在右边。
        处理的结果是将 node_idx 对应的局域张量分解为三个局域张量 u, s, v，其中 s 为对角矩阵，u中间的指标对应分组为左边的指标，v 中间的指标对应分组为右边的指标。
              L         R\n
            ┌────┐    ┌────┐\n
            │    │    │    │                     L                      R\n
            │    │    │    │\n
             │  │      │  │                     │  │                   │  │\n
           ┌─┴──┴──────┴──┴─┐                 ┌─┴──┴─┐   ┌──────┐    ┌─┴──┴─┐\n
           │                │                 │      │   │      │    │      │\n
        ───┤                ├──   ─────►   ───┤  u   ├───┤  s   ├────┤  v   ├──\n
           │                │                 │      │   │      │    │      │\n
           └────────────────┘                 └──────┘   └──────┘    └──────┘\n
        '''

        if node_idx < 0:
            node_idx = node_idx + len(self.node_list)
        node = self.node_list[node_idx]
        dims = node.shape
        if group_side == 'L':
            legs_L = [_ % node.dim() for _ in legs_group]
            legs_R = [_ for _ in range(1, node.dim())]
            for i in legs_L:
                legs_R.remove(i)
        elif group_side == 'R':
            legs_R = [_ % node.dim() for _ in legs_group]
            legs_L = [_ for _ in range(1, node.dim())]
            for i in legs_R:
                legs_L.remove(i)
        else:
            print("!WARNING! para: 'group_side' must be either 'L' or 'R'")
        dimL = 1
        for i in legs_L:
            dimL = dimL * dims[i]
        perm = [0] + legs_L[:] + legs_R[:]
        node = node.permute(perm)
        node = node + tc.randn(node.shape, dtype=node.dtype, device=node.device)*1e-10     
        u, s, v = tc.linalg.svd(node.reshape(dims[0], dimL, -1), full_matrices=False)
        # Deal with the near zero singular values.
        # If not may lead to infinite gradients!!!
        col = self.find_first_column_with_element_magnitude_less_than_e(s, 0)

        if if_trun:
            if self.chi == None:
                dc = col
            else:
                dc = min(self.chi, col)
            trunc_error = tc.sum(s[:, dc:], dim=1)
        else:
            dc = col
            trunc_error = 0
        self.trunc_error += trunc_error
        ####################### CHECK ##########################
        u = u[:, :, :dc].reshape([dims[0]] + [dims[i] for i in legs_L] + [dc])
        s = tc.diag_embed(s[:, :dc]).to(self.dtype)
        v = v[:, :dc, :].reshape([dims[0]] + [dc] + [dims[i] for i in legs_R])
        self.node_list = self.node_list[:node_idx] + [u, \
                                                        s, \
                                                        v] + self.node_list[node_idx+1:]
        self.history['split_node'].append([node_idx, legs_L[:], legs_R[:]])
        self.renew_graph()
        pass

    def flatten(self, node_pair:tuple):
        nodes = list(node_pair)
        node1 = self.node_list[nodes[0]]
        node2 = self.node_list[nodes[1]]
        left1 = list(i for i in range(node1.dim()))
        left2 = list(i for i in range(node2.dim()))
        legs_old_list = self.connect_graph.pop(node_pair)
        legs1 = []
        legs2 = []
        shape1 = []
        shape2 = []
        dim = 1
        for leg_pair in legs_old_list:
            legs1.append(leg_pair[0])
            legs2.append(leg_pair[1])
            left1.remove(leg_pair[0])
            left2.remove(leg_pair[1])
            dim = dim * node1.shape[leg_pair[0]]
        for i in left1:
            shape1.append(node1.shape[i])
        for j in left2:
            shape2.append(node2.shape[j])
        shape1 = shape1[:] + [dim]
        shape2 = [shape2[0], dim] + shape2[1:]
        perm1 = left1[:] + legs1[:]
        perm2 = [0] + legs2[:] + left2[1:]
        node1 = tc.permute(node1, perm1).reshape(shape1)
        node2 = tc.permute(node2, perm2).reshape(shape2)
        self.node_list[nodes[0]] = node1
        self.node_list[nodes[1]] = node2
        self.connect_graph[node_pair] = [(node1.dim()-1, 1)]
        # 处理connect_graph
        left1 = left1 + [None]
        left2 = [None] + left2
        keys = list(self.connect_graph.keys())[:]
        keys.remove(node_pair)
        for key in keys:
            old_legs = self.connect_graph[key]
            for j in range(2):
                if key[j] == node_pair[0]:
                    tmp = old_legs[:]
                    for i, leg_pair in enumerate(tmp):
                        leg_pair = list(leg_pair)
                        leg_pair[j] = left1.index(leg_pair[j])
                        old_legs[i] = tuple(leg_pair)
                elif key[j] == node_pair[1]:
                    tmp = old_legs[:]
                    for i, leg_pair in enumerate(tmp):
                        leg_pair = list(leg_pair)
                        leg_pair[j] = left2.index(leg_pair[j])
                        old_legs[i] = tuple(leg_pair)
            self.connect_graph[key] = old_legs


class TensorTrain(TensorNetwork):
    def __init__(self, tensors:list, length:int, phydim:int=2, center:int=-1, chi=None, initialize=True, device=tc.device('cpu'), dtype=tc.complex64):
        super().__init__(chi=chi, device=device, dtype=dtype)

        self.length = length
        self.phydim = phydim
        self.device = device
        self.dtype = dtype
        self.chi = chi
        self.center = center % self.length

        # center_reletive = self.center
        tensor = tensors[0]
        flags = [tensor.dim()-1]
        tensor = tc.unsqueeze(tensor, 0)
        tensor = tc.unsqueeze(tensor, -1)
        self.add_node(tensor, device=device, dtype=dtype)
        # if center_reletive > tensor.dim()-3:
        #     self.normalize_node(node_idx=-1, center=-1)
        #     center_reletive -= tensor.dim()-2
        # else:
        #     self.normalize_node(node_idx=-1, center=center_reletive)
        for tensor in tensors[1:]:
            flags.append(flags[-1] + tensor.dim())
            tensor = tc.unsqueeze(tensor, 0)
            tensor = tc.unsqueeze(tensor, -1)
            self.add_node(tensor, device=device, dtype=dtype)
            self.connect([-2, -1], [-1, 0])
        flags.pop()

        self.initialize(flags, if_trun=True)
        pass

    def initialize(self, flags, if_trun=True):
        for i in range(self.center):
            legs_L = [0, 1]
            if i in flags:
                self.merge_nodes((i, i+1))
            self.split_node(i, legs_L, group_side='L', if_trun=False)
            self.merge_nodes((i+1, i+2))
        
        for j in range(-1, -self.length + self.center, -1):
            legs_R = [-2, -1]
            if (j % self.length - 1) in flags:
                self.merge_nodes((j-1, j))
            self.split_node(j, legs_R, group_side='R', if_trun=False)
            self.merge_nodes((j-2, j-1))
        self.split_node(self.center, legs_group=[0,1], group_side='L', if_trun=if_trun)
        self.merge_nodes((self.center, self.center+1))
        self.merge_nodes((self.center, self.center+1))
        self.normalize()
    
    # def initialize__(self, if_trun=True):
    #     for i in range(self.center):
    #         if i == 0:
    #             legs_L = [0]
    #         else:
    #             legs_L = [0, 1]
    #         self.split_node(i, legs_L, group_side='L', if_trun=if_trun)
    #         self.merge_nodes((i+1, i+2))
    #     for j in range(-1, -self.length + self.center, -1):
    #         if j == -1:
    #             legs_R = [-1]
    #         else:
    #             legs_R = [-2, -1]
    #         self.split_node(j, legs_R, group_side='R', if_trun=if_trun)
    #         self.merge_nodes((j-2, j-1))
    #     self.normalize()

    def move_center_to(self, to_idx, if_trun=True):
        to_idx = to_idx % self.length
        if self.center < to_idx:
            move_path = list(i for i in range(self.center, to_idx))
            for i in move_path:
                self.split_node(i, legs_group=[-1], group_side='R', if_trun=if_trun)
                self.merge_nodes((i+1, i+2))
                self.merge_nodes((i+1, i+2))
                self.center = i+1
        else:
            move_path = list(i for i in range(self.center - self.length, to_idx - self.length, -1))
            for i in move_path:
                self.split_node(i, legs_group=[0], group_side='L', if_trun=if_trun)
                self.merge_nodes((i-1, i-2))
                self.merge_nodes((i-1, i-2))
                self.center = (i-1) % self.length

    # def act_n_body_gate(self, gate, pos:list):
    #     """
    #     gate: 作用的门，是形状为(2**n, 2**n)的矩阵（2也可以是phy_dim）
    #     pos: 连续的位置
    #     """
    #     n = len(pos)
    #     center = (n-1) // 2
    #     perm = [_ for _ in range(0, center)] + [_ for _ in range(n, n+center)] + [center]\
    #           + [_ for _ in range(center+1, n)] + [_ for _ in range(center+n+1, 2*n)] + [center+n]
    #     reshaped_gate = gate.reshape([self.phydim] * (2*n)).permute(perm).reshape(\
    #         [self.phydim**(2*center), self.phydim, self.phydim**(2*n-2*center-2), self.phydim])
    #     self.add_node(reshaped_gate, device=self.device, dtype=self.dtype)
    #     self.connect([-1, -1], [pos[center], 1])
    #     for i in range(center, 0, -1):
    #         Eye = tc.eye(self.phydim**(2*i), device=self.device, dtype=self.dtype).reshape([self.phydim]*(4*i))
    #         perm = [_ for _ in range(1, 2*i-1)] + [0] + [_ for _ in range(2*i, 4*i)] + [2*i-1]
    #         Eye = Eye.permute(perm).reshape([self.phydim**(2*i-2), self.phydim, self.phydim**(2*i), self.phydim])
    #         self.add_node(Eye, device=self.device, dtype=self.dtype)
    #         self.connect([-1, -1], [pos[i-1], 1])
    #         self.connect([-1, 2], [-2, 0])
    #         pass
    #     pass

    def act_one_body_gate(self, gate, pos:int):
        self.move_center_to(pos)
        self.add_node(gate, device=self.device, dtype=self.dtype)
        self.connect([pos, 1], [-1, 1])
        self.move_node(-1, pos)
        self.merge_nodes((pos, pos+1))
        self.permute_legs(pos, perm=[1, 0, 2])
        self.normalize()
        pass

    def act_two_body_gate(self, gate, pos):
        gr = tc.eye(self.phydim**2, device=self.device, dtype=self.dtype).reshape(
            [self.phydim]*4).permute(0, 2, 3, 1).reshape(self.phydim, self.phydim**2, self.phydim)
        gl = gate.reshape([self.phydim]*4).permute(0, 1, 3, 2).reshape(self.phydim, self.phydim**2, self.phydim)
        if self.center >= pos[1]:
            self.move_center_to(pos[1], if_trun=True)
        else:
            self.move_center_to(pos[0], if_trun=True)
        self.add_node(gl, device=self.device, dtype=self.dtype)
        self.add_node(gr, device=self.device, dtype=self.dtype)
        # 以后需要考虑pos[0], pos[1]不相邻的情况
        self.connect([-1, 1], [-2, 1])
        self.connect([pos[0], -2], [-2, -1])
        self.connect([pos[1], 1], [-1, -1])
        self.move_node(-2, pos[0]+1)
        self.merge_nodes((pos[0], pos[0]+1))
        self.move_node(-1, pos[1])
        self.merge_nodes((pos[1], pos[1]+1))
        # self.flatten(tuple(pos))
        i = pos[0]
        self.split_node(i, legs_group=[-1], group_side='R', if_trun=False)
        self.merge_nodes((i+1, i+2))
        self.merge_nodes((i+1, i+2))
        self.split_node(i+1, legs_group=[0], group_side='L', if_trun=True)
        self.merge_nodes((i, i+1))
        self.merge_nodes((i, i+1))
        self.center = i
        self.normalize()
        pass

    def get_norm(self):
        center_tn = self.node_list[self.center]
        if self.center not in [0, self.length-1]:
            norm = tc.einsum('ijk, ijk->', center_tn, center_tn.conj())
        else:
            norm = tc.einsum('ijk, ijk->', center_tn, center_tn.conj())
            # norm = tc.einsum('ij, ij->', center_tn, center_tn.conj())
        return norm.real
    
    def normalize(self):
        norm = self.get_norm()
        center_tn = self.node_list[self.center]
        self.node_list[self.center] = center_tn / tc.sqrt(norm)

class TensorTrain_pack(TensorNetwork_pack):
    def __init__(self, tensor_packs:list, length:int, phydim:int=2, center:int=-1, chi=None, device=tc.device('cpu'), dtype=tc.complex64, initialize=True):
        """
        tensor_packs: 包含多个局域张量的列表，列表中每一个元素(tensor)对应在该位置的n个局域张量，tensor[i].shape=[mps态的个数, 局域张量的形状]
        """
        super().__init__(chi=chi, device=device, dtype=dtype)

        self.length = length
        self.number = tensor_packs[0].shape[0]
        self.phydim = phydim
        self.device = device
        self.dtype = dtype
        self.chi = chi
        self.center = center % self.length
        self.trunc_error = tc.zeros(size=(tensor_packs[0].shape[0],), device=device, dtype=dtype)

        # center_reletive = self.center
        
        if initialize == True:
            # tensor = tensor_packs[0]
            # flags = [tensor.dim()-2]
            # tensor = tc.unsqueeze(tensor, 1)
            # tensor = tc.unsqueeze(tensor, -1)
            # self.add_node(tensor, device=device, dtype=dtype)
            #     center_reletive -= tensor.dim()-2
            # else:
            #     self.normalize_node(node_idx=-1, center=center_reletive)
            flags = []
            for i, tensor in enumerate(tensor_packs[:]):
                tensor = tc.unsqueeze(tensor, 1)
                tensor = tc.unsqueeze(tensor, -1)
                self.add_node(tensor, device=device, dtype=dtype)
                if i == 0:
                    flags.append(tensor.dim()-2)
                else:
                    flags.append(flags[-1] + tensor.dim()-1)
                    self.connect([-2, -1], [-1, 1])
            flags.pop()

            self.initialize(flags, if_trun=True)
        else:
            self.node_list = []
            for node in tensor_packs[:]:
                self.add_node(node.clone(), device=device, dtype=dtype)
                if len(self.node_list) > 1:
                    self.connect([-2, -1], [-1, 1])
        pass

    def initialize(self, flags, if_trun=True):
        for i in range(self.center):
            legs_L = [1, 2]
            if i in flags:
                self.merge_nodes((i, i+1))
            self.split_node(i, legs_L, group_side='L', if_trun=True)
            self.merge_nodes((i+1, i+2))
        
        for j in range(-1, -self.length + self.center, -1):
            legs_R = [-2, -1]
            if (j % self.length - 1) in flags:
                self.merge_nodes((j-1, j))
            self.split_node(j, legs_R, group_side='R', if_trun=True)
            self.merge_nodes((j-2, j-1))
        self.split_node(self.center, legs_group=[1,2], group_side='L', if_trun=if_trun)
        self.merge_nodes((self.center, self.center+1))
        self.merge_nodes((self.center, self.center+1))
        self.normalize()
    
    # def initialize__(self, if_trun=True):
    #     for i in range(self.center):
    #         if i == 0:
    #             legs_L = [0]
    #         else:
    #             legs_L = [0, 1]
    #         self.split_node(i, legs_L, group_side='L', if_trun=if_trun)
    #         self.merge_nodes((i+1, i+2))
    #     for j in range(-1, -self.length + self.center, -1):
    #         if j == -1:
    #             legs_R = [-1]
    #         else:
    #             legs_R = [-2, -1]
    #         self.split_node(j, legs_R, group_side='R', if_trun=if_trun)
    #         self.merge_nodes((j-2, j-1))
    #     self.normalize()

    def orthogonalize_part(self, start:int, end:int, if_trun=True):
        if start < end:
            for i in range(start, end):
                self.merge_nodes((i, i+1))
                self.split_node(i, legs_group=[1, 2], group_side='L', if_trun=if_trun)
                self.merge_nodes((i+1, i+2))
        elif start > end:
            for i in range(start, end, -1):
                self.merge_nodes((i-1, i))
                self.split_node(i-1, legs_group=[3, 4], group_side='R', if_trun=if_trun)
                self.merge_nodes((i-1, i))

    def move_center_to(self, to_idx, if_trun=True):
        to_idx = to_idx % self.length
        if type(self.center) == int:
            self.orthogonalize_part(start=self.center, end=to_idx, if_trun=if_trun)
            self.center = to_idx
        elif type(self.center) == list:
            affected_part = self.center[:] + [to_idx]
            affected_part.sort()
            self.orthogonalize_part(start=affected_part[0], end=to_idx, if_trun=if_trun)
            self.orthogonalize_part(start=affected_part[-1], end=to_idx, if_trun=if_trun)
            self.center = to_idx
        # if self.center < to_idx:
        #     move_path = list(i for i in range(self.center, to_idx))
        #     for i in move_path:
        #         self.split_node(i, legs_group=[-1], group_side='R', if_trun=if_trun)
        #         self.merge_nodes((i+1, i+2))
        #         self.merge_nodes((i+1, i+2))
        #         self.center = i+1
        # else:
        #     move_path = list(i for i in range(self.center - self.length, to_idx - self.length, -1))
        #     for i in move_path:
        #         self.split_node(i, legs_group=[1], group_side='L', if_trun=if_trun)
        #         self.merge_nodes((i-1, i-2))
        #         self.merge_nodes((i-1, i-2))
        #         self.center = (i-1) % self.length

    # def act_n_body_gate(self, gate, pos:list):
    #     """
    #     gate: 作用的门，是形状为(2**n, 2**n)的矩阵（2也可以是phy_dim）
    #     pos: 连续的位置
    #     """
    #     n = len(pos)
    #     center = (n-1) // 2
    #     perm = [_ for _ in range(0, center)] + [_ for _ in range(n, n+center)] + [center]\
    #           + [_ for _ in range(center+1, n)] + [_ for _ in range(center+n+1, 2*n)] + [center+n]
    #     reshaped_gate = gate.reshape([self.phydim] * (2*n)).permute(perm).reshape(\
    #         [self.phydim**(2*center), self.phydim, self.phydim**(2*n-2*center-2), self.phydim])
    #     self.add_node(reshaped_gate, device=self.device, dtype=self.dtype)
    #     self.connect([-1, -1], [pos[center], 1])
    #     for i in range(center, 0, -1):
    #         Eye = tc.eye(self.phydim**(2*i), device=self.device, dtype=self.dtype).reshape([self.phydim]*(4*i))
    #         perm = [_ for _ in range(1, 2*i-1)] + [0] + [_ for _ in range(2*i, 4*i)] + [2*i-1]
    #         Eye = Eye.permute(perm).reshape([self.phydim**(2*i-2), self.phydim, self.phydim**(2*i), self.phydim])
    #         self.add_node(Eye, device=self.device, dtype=self.dtype)
    #         self.connect([-1, -1], [pos[i-1], 1])
    #         self.connect([-1, 2], [-2, 0])
    #         pass
    #     pass

    def act_one_body_gate(self, gate, pos:int):
        # not finished
        self.move_center_to(pos)
        self.add_node(gate, device=self.device, dtype=self.dtype)
        self.connect([pos, 2], [-1, 1])
        self.move_node(-1, pos)
        self.merge_nodes((pos, pos+1), is_gate=(True, False))
        self.permute_legs(pos, cycle=[1, 2])
        self.normalize()
        pass

    def act_two_body_gate(self, gate, pos):
        '''
        
        '''
        gr = tc.eye(self.phydim**2, device=self.device, dtype=self.dtype).reshape(
            [self.phydim]*4).permute(0, 2, 3, 1).reshape(self.phydim, self.phydim**2, self.phydim)
        gl = gate.reshape([self.phydim]*4).permute(0, 1, 3, 2).reshape(self.phydim, self.phydim**2, self.phydim)
        if self.center >= pos[1]:
            self.move_center_to(pos[1], if_trun=True)
        else:
            self.move_center_to(pos[0], if_trun=True)
        self.add_node(gl, device=self.device, dtype=self.dtype)
        self.add_node(gr, device=self.device, dtype=self.dtype)
        # 以后需要考虑pos[0], pos[1]不相邻的情况
        self.connect([-1, 1], [-2, 1])
        self.connect([pos[0], -2], [-2, -1])
        self.connect([pos[1], 2], [-1, -1])
        self.move_node(-2, pos[0]+1)
        self.merge_nodes((pos[0], pos[0]+1), is_gate=(False, True))
        self.move_node(-1, pos[1])
        self.merge_nodes((pos[1], pos[1]+1), is_gate=(True, False))
        # self.flatten(tuple(pos))
        i = pos[0]
        self.merge_nodes((i, i+1))
        self.split_node(i, legs_group=[1, 2], group_side='L', if_trun=True)
        self.merge_nodes((i+1, i+2))
        # self.split_node(i, legs_group=[-1], group_side='R', if_trun=False)
        # self.merge_nodes((i+1, i+2))
        # self.merge_nodes((i+1, i+2))
        # self.split_node(i+1, legs_group=[1], group_side='L', if_trun=True)
        # self.merge_nodes((i, i+1))
        # self.merge_nodes((i, i+1))
        self.center = i+1
        self.normalize()
        return self

    def act_n_body_gate(self, gate, pos:list):
        def fill_seq(pos:list):
            delta_pos = list(i for i in range(pos[0], pos[-1]+1))
            for i in pos:
                delta_pos.remove(i)
            return delta_pos
        def step_function(i:int, pos:list):
            f = lambda x: 0 if x < 0 else 1
            y = 0
            for j in pos[1:]:
                y = y + f(i - j)
            return y

        if type(self.center) == int:
            affected_pos = pos[:] + [self.center]
            affected_pos.sort()
            self.center = [affected_pos[0], affected_pos[-1]]
        elif type(self.center) == list:
            affected_pos = pos[:] + self.center[:]
            affected_pos.sort()
            self.center = [affected_pos[0], affected_pos[-1]]
        else:
            print('error: TensorTrain_pack.center should be int or list')
        # affected_pos = pos[:] + next_pos[:]
        # affected_pos.sort()
        # if self.center < affected_pos[0]:
        #     self.move_center_to(to_idx=affected_pos[0])
        # elif self.center > affected_pos[-1]:
        #     self.move_center_to(to_idx=affected_pos[-1])

        gate_list = n_body_gate_to_mpo(gate, n=len(pos), phydim=self.phydim, device=self.device, dtype=self.dtype)
        # print(len(gate_list))
        delta_dim_list = [gate_list[step_function(i, pos)].shape[-1] for i in range(pos[0], pos[-1]+1)]
        delta_pos = fill_seq(pos)
        # print('pos=', pos)
        self.add_node(gate_list[0].squeeze(), site=pos[0]+1, device=self.device, dtype=self.dtype)
        self.connect([pos[0], 2], [pos[0]+1, 1])
        self.merge_nodes((pos[0], pos[0]+1), is_gate=(False, True))
        ############## NOT SUITBLE FOR MULTI-BONDS BETWEEN TWO NODES ############
        self.permute_legs(pos[0], cycle=[i for i in range(2, self.node_list[pos[0]].dim())])
        gate_idx = 1
        for i in range(pos[0]+1, pos[-1]):
            # print(i)
            if i in pos:
                self.add_node(gate_list[gate_idx], site=i+1, device=self.device, dtype=self.dtype)
                self.connect([i, 2], [i+1, 2])
                self.connect([i-1, -2], [i+1, 0])
                self.merge_nodes((i, i+1), is_gate=(False, True))
                self.flatten((i-1, i))
                # self.permute_legs(i, cycle=[2, 3, 4, 5])
                self.permute_legs(i, cycle=[2, 3, 4])
                gate_idx += 1
                pass
            elif i in delta_pos:
                delta_dim = delta_dim_list[i]
                delta = tc.einsum('il, jk -> ijkl', tc.eys(delta_dim, device=self.device, dtype=self.dtype), tc.eys(self.phydim, device=self.device, dtype=self.dtype))
                self.add_node(delta, site=i+1, device=self.device, dtype=self.dtype)
                self.connect([i, 2], [i+1, 1])
                self.connect([i-1, -2], [i+1, 0])
                self.merge_nodes((i, i+1), is_gate=(False, False))
                self.flatten((i-1, i))
                self.permute_legs(i, cycle=[2, 3, 4])
                # self.permute_legs(i, cycle=[2, 3, 4, 5])
        if len(pos) > 1:
            self.add_node(gate_list[-1].squeeze(), site=pos[-1]+1, device=self.device, dtype=self.dtype)
            self.connect([pos[-1], -2], [pos[-1]+1, 2])
            self.connect([pos[-1]-1, -2], [pos[-1]+1, 0])
            self.merge_nodes((pos[-1], pos[-1]+1), is_gate=(False, True))
            self.flatten((pos[-1]-1, pos[-1]))
            self.permute_legs(pos[-1], cycle=[2, 3])

        # for i in range(affected_pos[0], next_pos[0]):
        #     self.merge_nodes((i, i+1))
        #     self.split_node(i, legs_group=[1, 2], group_side='L', if_trun=True)
        #     self.merge_nodes((i+1, i+2))
        # for i in range(affected_pos[-1], next_pos[-1], -1):
        #     self.merge_nodes((i-1, i))
        #     self.split_node(i-1, legs_group=[2, 3], group_side='R', if_trun=True)
        #     self.merge_nodes((i-1, i))

        # if set_center < pos[0]:
        #     for i in range(pos[-1], set_center, -1):
        #         self.merge_nodes((i-1, i))
        #         self.split_node(i-1, legs_group=[2, 3], group_side='R', if_trun=True)
        #         self.merge_nodes((i-1, i))
        # elif set_center > pos[-1]:
        #     for i in range(pos[0], set_center):
        #         self.merge_nodes((i, i+1))
        #         self.split_node(i, legs_group=[1, 2], group_side='L', if_trun=True)
        #         self.merge_nodes((i+1, i+2))
        # else:
        #     for i in range(pos[0], set_center):
        #         self.merge_nodes((i, i+1))
        #         self.split_node(i, legs_group=[1, 2], group_side='L', if_trun=True)
        #         self.merge_nodes((i+1, i+2))

        #     for i in range(pos[-1], set_center, -1):
        #         self.merge_nodes((i-1, i))
        #         self.split_node(i-1, legs_group=[2, 3], group_side='R', if_trun=True)
        #         self.merge_nodes((i-1, i))

        # self.center = set_center
        return self

    def act_n_body_gate_sequence(self, gate, pos_list, set_center, if_trun=True):
        # def gen_center_sequence(pos_list):
        #     center_list = []
        if self.center < pos_list[0][0]:
            self.move_center_to(pos_list[0][0], if_trun=if_trun)
        elif self.center > pos_list[0][-1]:
            self.move_center_to(pos_list[0][-1], if_trun=if_trun)
        self.act_n_body_gate(gate, pos_list[0])
        self.center = pos_list[0][:]

        for i, pos in enumerate(pos_list[1:]):
            # print('-'*10)
            # print('the ', i, '-th gate:')
            # print('pos:', pos)
            # print('center:', self.center)
            affected_pos = self.center[:] + pos[:]
            affected_pos.sort()
            # print('affected_pos:', affected_pos)
            self.orthogonalize_part(start=affected_pos[0], end=pos[0], if_trun=if_trun)
            self.orthogonalize_part(start=affected_pos[-1], end=pos[-1], if_trun=if_trun)
            self.act_n_body_gate(gate, pos)
            self.center = pos[:]
        self.move_center_to(set_center, if_trun=if_trun)
        return self

    def get_norm(self):
        center_tn = self.node_list[self.center]
        # if self.center not in [0, self.length-1]:
        norm = tc.einsum('nijk, nijk->n', center_tn.detach(), center_tn.detach().conj())
        # else:
            # norm = tc.einsum('nijk, nijk->n', center_tn, center_tn.conj())
            # norm = tc.einsum('ij, ij->', center_tn, center_tn.conj())
        return norm.real
    
    def normalize(self):
        norm = self.get_norm()
        center_tn = self.node_list[self.center]
        # 创建一个新的tensor，先复制原有的self.node_list[self.center]
        new_tensor = self.node_list[self.center].clone()

        # 然后对new_tensor进行赋值操作
        for i in range(norm.numel()):
            # self.node_list[self.center][i] = center_tn[i] / tc.sqrt(norm)[i]
            new_tensor[i] = center_tn[i] / tc.sqrt(norm)[i]
        # 最后，将新的tensor赋值回self.node_list[self.center]
        self.node_list[self.center] = new_tensor
        # self.node_list[self.center] = center_tn / tc.sqrt(norm)

from torch.utils.data import Dataset, DataLoader
class MPS_Dataset(Dataset, TensorTrain_pack):
    def __init__(self, data:TensorTrain_pack) -> None:
        super(Dataset, self).__init__(data.node_list, length=data.length,\
                                        phydim=data.phydim, center=data.center,\
                                        chi=data.chi, device=data.device,\
                                        dtype=data.dtype, initialized=True)
        # self.data = self.node_list
    
    def __len__(self):
        return self.node_list[0].shape[0]
    
    def __getitem__(self, index) -> list:
        item = list()
        for i in range(self.length):
            item.append(self.node_list[i][index])
        return item


def combine_mps_packs(mps1:TensorTrain_pack, mps2:TensorTrain_pack):
    new_mps = copy_from_mps_pack(mps1)
    mps2.move_center_to(mps1.center)
    for i in range(mps2.length):
        new_mps.node_list[i] = pad_and_cat([new_mps.node_list[i], mps2.node_list[i]], dim=0)
    new_mps.trunc_error = pad_and_cat([new_mps.trunc_error, mps2.trunc_error], dim=0)
    return new_mps

def slice_mps_pack(mps:TensorTrain_pack, section:int):
    new_mps = copy_from_mps_pack(mps)
    for i in range(mps.length):
        new_mps.node_list[i] = new_mps.node_list[i][:section]
    if new_mps.trunc_error != 0:
        new_mps.trunc_error = new_mps.trunc_error[:section]
    return new_mps

def rand_prod_mps_pack(number, length, chi, phydim=2, device=tc.device('cpu'), dtype=tc.complex64, **kwargs):
    states = [tc.rand([number, 1, phydim, 1], dtype=dtype, device=device) for _ in range(length)]
    for i in range(len(states)):
        site = states[i]
        norm = tc.sqrt(tc.einsum('naib, naib->nab', site, site.conj()))
        site = site / norm.unsqueeze(2).broadcast_to(size=site.shape)
        states[i] = site
    mps = TensorTrain_pack(states, length=length, phydim=phydim, center=-1, chi=chi, device=device, dtype=dtype, initialize=False)
    return mps

def rand_mps_pack(number, length, chi, phydim=2, device=tc.device('cpu'), dtype=tc.complex64):
    local_tensors = [tc.rand([number, 1, phydim, chi], dtype=dtype, device=device)]\
          + [tc.rand([number, chi, phydim, chi], dtype=dtype, device=device) for _ in range(length-2)]\
          + [tc.rand([number, chi, phydim, 1], dtype=dtype, device=device)]
    mps = TensorTrain_pack(local_tensors, length=length, phydim=phydim, center=length//2, chi=chi, device=device, dtype=dtype, initialize=False)
    mps.initialize(flags=[i for i in range(length)])
    return mps

def inner_mps_pack(mps1:TensorTrain_pack, mps2:TensorTrain_pack):
    length = len(mps1.node_list)
    mps1_nodes = mps1.node_list
    mps2_nodes = mps2.node_list
    result = TensorNetwork_pack(chi=None, device=mps1.device, dtype=mps1.dtype)
    result.add_node(mps1_nodes[0].conj())
    result.add_node(mps2_nodes[0])
    result.connect([0, 2], [1, 2])
    result.connect([0, 1], [1, 1])
    result.merge_nodes((0, 1))
    for i in range(1, length):
        result.add_node(mps1_nodes[i].conj())
        result.add_node(mps2_nodes[i])
        result.connect([-1, 2], [-2, 2])
        result.connect([-2, 1], [-3, 1])
        result.connect([-1, 1], [-3, 2])
        result.merge_nodes((-1, -2))
        # result.flatten((0, 1))
        result.merge_nodes((0, 1))
    return tc.squeeze(result.node_list[0])

def one_body_obs_from_mps_pack(mps:TensorTrain_pack, obs:list):
    obs_exp = tc.zeros((mps.number, len(obs), mps.length), device=mps.device, dtype=mps.dtype)
    obs_num = len(obs)
    for i in range(mps.length):
        mps.move_center_to(i)
        center_tensor = mps.node_list[i]
        for j in range(obs_num):
            ob_exp = tc.einsum('nijk, jl, nilk->n', center_tensor.conj(), obs[j], center_tensor)
            obs_exp[:, j, i] = ob_exp
    return obs_exp

def multi_mags_from_mps_pack(mps:TensorTrain_pack, spins:list):
    # mps_ = deepcopy(mps)
    mags = tc.zeros((mps.node_list[0].shape[0], len(spins), mps.length), device=mps.device, dtype=mps.dtype)
    num_spin = len(spins)
    for i in range(mps.length):
        mps.move_center_to(i)
        center_tensor = mps.node_list[i]
        for s in range(num_spin):
            hz = tc.einsum('nijk, jl, nilk->n', center_tensor.conj(), spins[s], center_tensor)
            mags[:, s, i] = hz
    return mags

def copy_from_mps_pack(mps:TensorTrain_pack):
    new_node_list = []
    for node in mps.node_list:
        new_node_list.append(node.detach().clone())
    new_mps = TensorTrain_pack(new_node_list, length=mps.length, phydim=mps.phydim, center=mps.phydim, chi=mps.chi, device=mps.device, dtype=mps.dtype, initialize=False)
    new_mps.connect_graph = deepcopy(mps.connect_graph)
    return new_mps

def tensor2mps_pack(states:tc.Tensor, chi:int):
    length = states.dim() - 1
    phydim = states.shape[-1]
    states = states.unsqueeze(1)
    states = states.unsqueeze(-1)
    mps_pack = TensorTrain_pack(tensor_packs=[states], length=length, phydim=phydim, center=-1, chi=chi, device=states.device, dtype=states.dtype, initialize=False)

def n_body_gate_to_mpo(gate, n:int, phydim=2, device=tc.device('cpu'), dtype=tc.complex64):
    '''
    The shape of gate should be [2, 2, ..., 2, 2]
    '''
    # n = gate.dim() // 2
    center_gate = gate.unsqueeze(0)
    center_gate = center_gate.unsqueeze(-1)
    left_gate_list = []
    right_gate_list = []
    while n > 2:
        shape = list(center_gate.shape)
        left_dim = shape[0]
        right_dim = shape[-1]
        left_gate = tc.eye(phydim * left_dim * phydim, device=device, dtype=dtype)
        left_gate = left_gate.reshape([phydim, left_dim, phydim, phydim**2 * left_dim])
        left_gate = left_gate.permute(dims=[1, 0, 2, 3])
        right_gate = tc.eye(phydim**2 * right_dim, device=device, dtype=dtype)
        right_gate = right_gate.reshape([phydim**2 * left_dim, phydim, right_dim, phydim])
        right_gate = right_gate.permute(dims=[0, 1, 3, 2])
        center_gate = center_gate.permute([1, 0, n+1] + [i for i in range(2, n)] + [i for i in range(n+2, 2*n)] + [n, 2*n+1, 2*n])
        center_gate = center_gate.reshape([phydim**2 * left_dim] + shape[2:n] + shape[n+2:2*n] + [phydim**2 * right_dim])
        left_gate_list = left_gate_list + [left_gate]
        right_gate_list = [right_gate] + right_gate_list
        n = n - 2
    if n == 2:
        shape = list(center_gate.shape)
        left_dim = shape[0]
        right_dim = shape[-1]
        left_gate = tc.eye(phydim * left_dim * phydim, device=device, dtype=dtype)
        left_gate = left_gate.reshape([phydim, left_dim, phydim, phydim**2 * left_dim])
        left_gate = left_gate.permute(dims=[1, 0, 2, 3])
        left_gate_list = left_gate_list + [left_gate]
        # print('center_gate.shape=', center_gate.shape)
        center_gate = center_gate.permute([1, 0, 3, 2, 4, 5])
        center_gate = center_gate.reshape([phydim**2 * left_dim, phydim, phydim, right_dim])
    gate_list = left_gate_list + [center_gate] + right_gate_list
    return gate_list

if __name__ == '__main__':
    t1 = time.time()
    length = 4
    t = [tc.rand([5, 2], dtype=tc.complex64) for _ in range(length)]
    print(len(t))
    # t[0] = 1
    # t = t.reshape([2]*20)
    mps = TensorTrain_pack(t, length=length, phydim=2, center=0, chi=3, device=tc.device('cpu'), dtype=tc.complex64)
    print(mps.get_norm())
    for i in mps.node_list:
        print(i.shape)
        print()
    # print(mps.connect_graph)
    mps.act_one_body_gate(gate=tc.rand([2, 2], dtype=tc.complex64), pos=1)
    print(mps.connect_graph)
    gate = tc.rand([4, 4], dtype=tc.complex64)
    mps.act_two_body_gate(gate, pos = [1, 2])
    for i, t in enumerate(mps.node_list):
        print(f"site{i} shape:", t.shape)
    print(mps.connect_graph)
    print(mps.get_norm())
    print('center is ', mps.center)
    a0 = mps.node_list[-1]
    print(tc.einsum('nijk, nljk->nil', a0, a0.conj()))
    print(time.time() - t1)