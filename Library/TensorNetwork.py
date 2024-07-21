from platform import node
import time
from turtle import left
from debugpy import connect
import torch as tc
import numpy as np
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
        self.node_list = self.node_list[:node_idx] + [u.to(device=self.device, dtype=self.dtype), \
                                                        s.to(device=self.device, dtype=self.dtype), \
                                                        v.to(device=self.device, dtype=self.dtype)] + self.node_list[node_idx+1:]
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

    def get_merged_node(self, node_pair:tuple, is_gate:tuple=(False, False)):
        node_pair = list(node_pair)
        for i in range(2):
            if node_pair[i] < 0:
                node_pair[i] = len(self.node_list) + node_pair[i]
        node_pair = tuple(sorted(node_pair))
        node1 = self.node_list[node_pair[0]]
        node2 = self.node_list[node_pair[1]]
        n1_left_legs = [_ for _ in range(0, len(node1.shape))]
        n2_left_legs = [_ for _ in range(0, len(node2.shape))]
        
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
        elif is_gate[0] == True and is_gate[1] == False:
            perm_1 = n1_left_legs+n1_merge_legs
            perm_2 = [n2_left_legs[0]]+n2_merge_legs+n2_left_legs[1:]
            shape_1 = [-1, dim1]
            shape_2 = [node2.shape[0], dim2, -1]
            einsum_str = 'ij,njk->nik'
            new_shape = [node2.shape[0]]+[node1.shape[i] for i in n1_left_legs]+[node2.shape[j] for j in n2_left_legs[1:]]
            n1_left_legs = [None] + n1_left_legs[:]
            n2_left_legs = n2_left_legs[1:]
        elif is_gate[0] == False and is_gate[1] == True:
            perm_1 = n1_left_legs+n1_merge_legs
            perm_2 = n2_merge_legs+n2_left_legs
            shape_1 = [node1.shape[0], -1, dim1]
            shape_2 = [dim2, -1]
            einsum_str = 'nij,jk->nik'
            new_shape = [node1.shape[i] for i in n1_left_legs]+[node2.shape[j] for j in n2_left_legs]
        elif is_gate[0] == False and is_gate[1] == False:
            perm_1 = n1_left_legs+n1_merge_legs
            perm_2 = [n2_left_legs[0]]+n2_merge_legs+n2_left_legs[1:]
            shape_1 = [node1.shape[0], -1, dim1]
            shape_2 = [node2.shape[0], dim2, -1]
            einsum_str = 'nij,njk->nik'
            new_shape = [node1.shape[i] for i in n1_left_legs]+[node2.shape[j] for j in n2_left_legs[1:]]
            n2_left_legs = n2_left_legs[1:]

        node1_ = node1.permute(perm_1).reshape(shape_1)
        node2_ = node2.permute(perm_2).reshape(shape_2)
        new_node_ = tc.einsum(einsum_str, node1_, node2_)
        new_node = new_node_.reshape(new_shape)
        # new_node = tc.tensordot(node1, node2, dims=(n1_merge_legs, n2_merge_legs))
        return new_node, n1_left_legs, n2_left_legs

    def merge_nodes(self, node_pair:tuple, is_gate:tuple=(False, False)):
        node_pair = list(node_pair)
        for i in range(2):
            if node_pair[i] < 0:
                node_pair[i] = len(self.node_list) + node_pair[i]
        node_pair = tuple(sorted(node_pair))
        new_node, n1_left_legs, n2_left_legs = self.get_merged_node(node_pair, is_gate=is_gate)
        self.node_list = self.node_list[:node_pair[0]] + [new_node] + self.node_list[node_pair[0]+1:node_pair[1]] + self.node_list[node_pair[1]+1:]
        self.history['merge_nodes'].append([node_pair, n1_left_legs, n2_left_legs])
        self.renew_graph()

    def permute_legs(self, node_idx:int, perm:list):
        '''
        perm: 局域张量指标交换后的顺序，注意从1开始
        '''
        node = self.node_list[node_idx]
        self.node_list[node_idx] = tc.permute(node, [0]+perm)
        keys = list(self.connect_graph.keys())[:]
        for key in keys:
            for i in range(len(key)):
                if key[i] == node_idx:
                    connect_legs = self.connect_graph[key]
                    for j, legs in enumerate(connect_legs):
                        tmp = list(legs)
                        tmp[i] = perm[legs[i] - 1]
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
        u, s, v = tc.linalg.svd(node.reshape(dims[0], dimL, -1), full_matrices=False)
        if if_trun:
            if self.chi == None:
                dc = s.shape[1]
            else:
                dc = min(self.chi, s.shape[1])
            trunc_error = tc.sum(s[:, dc:], dim=1)
        else:
            dc = s.shape[1]
            trunc_error = 0
        self.trunc_error += trunc_error
        u = u[:, :, :dc].reshape([dims[0]] + [dims[i] for i in legs_L] + [dc])
        s = tc.diag_embed(s[:, :dc])
        v = v[:, :dc, :].reshape([dims[0]] + [dc] + [dims[i] for i in legs_R])
        self.node_list = self.node_list[:node_idx] + [u.to(device=self.device, dtype=self.dtype), \
                                                        s.to(device=self.device, dtype=self.dtype), \
                                                        v.to(device=self.device, dtype=self.dtype)] + self.node_list[node_idx+1:]
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
    def __init__(self, tensors:list, length:int, phydim:int=2, center:int=-1, chi=None, device=tc.device('cpu'), dtype=tc.complex64):
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

    def act_n_body_gate(self, gate, pos:list):
        """
        gate: 作用的门，是形状为(2**n, 2**n)的矩阵（2也可以是phy_dim）
        pos: 连续的位置
        """
        n = len(pos)
        center = (n-1) // 2
        perm = [_ for _ in range(0, center)] + [_ for _ in range(n, n+center)] + [center]\
              + [_ for _ in range(center+1, n)] + [_ for _ in range(center+n+1, 2*n)] + [center+n]
        reshaped_gate = gate.reshape([self.phydim] * (2*n)).permute(perm).reshape(\
            [self.phydim**(2*center), self.phydim, self.phydim**(2*n-2*center-2), self.phydim])
        self.add_node(reshaped_gate, device=self.device, dtype=self.dtype)
        self.connect([-1, -1], [pos[center], 1])
        for i in range(center, 0, -1):
            Eye = tc.eye(self.phydim**(2*i), device=self.device, dtype=self.dtype).reshape([self.phydim]*(4*i))
            perm = [_ for _ in range(1, 2*i-1)] + [0] + [_ for _ in range(2*i, 4*i)] + [2*i-1]
            Eye = Eye.permute(perm).reshape([self.phydim**(2*i-2), self.phydim, self.phydim**(2*i), self.phydim])
            self.add_node(Eye, device=self.device, dtype=self.dtype)
            self.connect([-1, -1], [pos[i-1], 1])
            self.connect([-1, 2], [-2, 0])
            pass
        pass

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
        self.flatten(tuple(pos))
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
    def __init__(self, tensor_packs:list, length:int, phydim:int=2, center:int=-1, chi=None, device=tc.device('cpu'), dtype=tc.complex64):
        """
        tensor_packs: 包含多个局域张量的列表，列表中每一个元素(tensor)对应在该位置的n个局域张量，tensor[i].shape=[mps态的个数, 局域张量的形状]
        """
        super().__init__(chi=chi, device=device, dtype=dtype)

        self.length = length
        self.phydim = phydim
        self.device = device
        self.dtype = dtype
        self.chi = chi
        self.center = center % self.length

        # center_reletive = self.center
        tensor = tensor_packs[0]
        flags = [tensor.dim()-2]
        tensor = tc.unsqueeze(tensor, 1)
        tensor = tc.unsqueeze(tensor, -1)
        self.add_node(tensor, device=device, dtype=dtype)
        # if center_reletive > tensor.dim()-3:
        #     self.normalize_node(node_idx=-1, center=-1)
        #     center_reletive -= tensor.dim()-2
        # else:
        #     self.normalize_node(node_idx=-1, center=center_reletive)
        for tensor in tensor_packs[1:]:
            flags.append(flags[-1] + tensor.dim()-1)
            tensor = tc.unsqueeze(tensor, 1)
            tensor = tc.unsqueeze(tensor, -1)
            self.add_node(tensor, device=device, dtype=dtype)
            self.connect([-2, -1], [-1, 1])
        flags.pop()

        self.initialize(flags, if_trun=True)
        pass

    def initialize(self, flags, if_trun=True):
        for i in range(self.center):
            legs_L = [1, 2]
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
                self.split_node(i, legs_group=[1], group_side='L', if_trun=if_trun)
                self.merge_nodes((i-1, i-2))
                self.merge_nodes((i-1, i-2))
                self.center = (i-1) % self.length

    def act_n_body_gate(self, gate, pos:list):
        """
        gate: 作用的门，是形状为(2**n, 2**n)的矩阵（2也可以是phy_dim）
        pos: 连续的位置
        """
        n = len(pos)
        center = (n-1) // 2
        perm = [_ for _ in range(0, center)] + [_ for _ in range(n, n+center)] + [center]\
              + [_ for _ in range(center+1, n)] + [_ for _ in range(center+n+1, 2*n)] + [center+n]
        reshaped_gate = gate.reshape([self.phydim] * (2*n)).permute(perm).reshape(\
            [self.phydim**(2*center), self.phydim, self.phydim**(2*n-2*center-2), self.phydim])
        self.add_node(reshaped_gate, device=self.device, dtype=self.dtype)
        self.connect([-1, -1], [pos[center], 1])
        for i in range(center, 0, -1):
            Eye = tc.eye(self.phydim**(2*i), device=self.device, dtype=self.dtype).reshape([self.phydim]*(4*i))
            perm = [_ for _ in range(1, 2*i-1)] + [0] + [_ for _ in range(2*i, 4*i)] + [2*i-1]
            Eye = Eye.permute(perm).reshape([self.phydim**(2*i-2), self.phydim, self.phydim**(2*i), self.phydim])
            self.add_node(Eye, device=self.device, dtype=self.dtype)
            self.connect([-1, -1], [pos[i-1], 1])
            self.connect([-1, 2], [-2, 0])
            pass
        pass

    def act_one_body_gate(self, gate, pos:int):
        # not finished
        self.move_center_to(pos)
        self.add_node(gate, device=self.device, dtype=self.dtype)
        self.connect([pos, 2], [-1, 1])
        self.move_node(-1, pos)
        self.merge_nodes((pos, pos+1), is_gate=(True, False))
        self.permute_legs(pos, perm=[2, 1, 3])
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
        self.connect([pos[1], 2], [-1, -1])
        self.move_node(-2, pos[0]+1)
        self.merge_nodes((pos[0], pos[0]+1), is_gate=(False, True))
        self.move_node(-1, pos[1])
        self.merge_nodes((pos[1], pos[1]+1), is_gate=(True, False))
        self.flatten(tuple(pos))
        i = pos[0]
        self.split_node(i, legs_group=[-1], group_side='R', if_trun=False)
        self.merge_nodes((i+1, i+2))
        self.merge_nodes((i+1, i+2))
        self.split_node(i+1, legs_group=[1], group_side='L', if_trun=True)
        self.merge_nodes((i, i+1))
        self.merge_nodes((i, i+1))
        self.center = i
        self.normalize()
        pass

    def get_norm(self):
        center_tn = self.node_list[self.center]
        if self.center not in [0, self.length-1]:
            norm = tc.einsum('nijk, nijk->n', center_tn, center_tn.conj())
        else:
            norm = tc.einsum('nijk, nijk->n', center_tn, center_tn.conj())
            # norm = tc.einsum('ij, ij->', center_tn, center_tn.conj())
        return norm.real
    
    def normalize(self):
        norm = self.get_norm()
        center_tn = self.node_list[self.center]
        for i in range(norm.numel()):
            self.node_list[self.center][i] = center_tn[i] / tc.sqrt(norm)[i]
        # self.node_list[self.center] = center_tn / tc.sqrt(norm)

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