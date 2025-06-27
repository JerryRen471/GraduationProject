import math
import os
import time

import numpy as np
import torch as tc
from matplotlib import pyplot as plt

import torchvision as torchvision
import copy
from termcolor import cprint
import torchvision.transforms as transforms
from inspect import stack
# import cv2
from datetime import datetime
import random


def binary_strings(num):
    s = list()
    length = len(str(bin(num-1))[2:])
    for n in range(num):
        b = str(bin(n))[2:]
        l0 = len(b)
        if length > l0:
            b = ''.join([('0' * (length-l0)), b])
        s.append(b)
    return s


def combine_dicts(dic_def, dic_new, deep_copy=False):
    # dic_def中的重复key值将被dic_new覆盖
    import copy
    if dic_new is None:
        return dic_def
    if deep_copy:
        return dict(copy.deepcopy(dic_def), **copy.deepcopy(dic_new))
    else:
        return dict(dic_def, **dic_new)


def convert_nums_to_abc(nums, n0=0):
    s = ''
    n0 = n0 + 97
    for m in nums:
        s += chr(m + n0)
    return s


def choose_device(n=0):
    if n == 'cpu':
        return 'cpu'
    else:
        if tc.cuda.is_available():
            if n is None:
                return tc.device("cuda:0")
            elif type(n) is int:
                return tc.device("cuda:"+str(n))
            else:
                return tc.device("cuda"+str(n)[4:])
        else:
            return tc.device("cpu")


def empty_list(num, content=None):
    return [content] * num


def find_indexes_value_in_list(x, value):
    return [n for n, v in enumerate(x) if v == value]


def fprint(content, file=None, print_screen=True, append=True):
    if file is None:
        file = './record.log'
    if append:
        way = 'ab'
    else:
        way = 'wb'
    with open(file, way, buffering=0) as log:
        log.write((content + '\n').encode(encoding='utf-8'))
    if print_screen:
        print(content)


def indexes_eq2einsum_eq(indexes):
    eq = convert_nums_to_abc(indexes[0])
    for n in range(1, len(indexes)-1):
        eq += (',' + convert_nums_to_abc(indexes[n]))
    eq += ('->' + convert_nums_to_abc(indexes[-1]))
    return eq


def list_eq2einsum_eq(eq):
    # 将list表示的equation转化为einsum函数的equation
    # list中的数字不能超过25！！！
    # 例如[[0, 1], [0, 2], [1, 2]] 转为 'ab,ac->bc'
    # 例如[[0, 1], [0, 1], []] 转为 'ab,ab->'
    length = len(eq)
    eq_str = ''
    for n in range(length-1):
        tmp = [chr(m+97) for m in eq[n]]
        eq_str = eq_str + ''.join(tmp) + ','
    eq_str = eq_str[:-1] + '->'
    tmp = [chr(m+97) for m in eq[-1]]
    return eq_str + ''.join(tmp)


def load(path_file, names=None, device='cpu'):
    if os.path.isfile(path_file):
        if names is None:
            data = tc.load(path_file)
            return data
        else:
            tmp = tc.load(path_file, map_location=device)
            if type(names) is str:
                data = tmp[names]
                return data
            elif type(names) in [tuple, list]:
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                return tuple(data)
            else:
                return None
    else:
        return None


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plot(x, *y, marker='s'):
    if type(x) is tc.Tensor:
        if x.device != 'cpu':
            x = x.cpu()
        x = x.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if len(y) > 0.5:
        for y0 in y:
            if type(y0) is tc.Tensor:
                if y0.device != 'cpu':
                    y0 = y0.cpu()
                y0 = y0.numpy()
            ax.plot(x, y0, marker=marker)
    else:
        ax.plot(x, marker=marker)
    plt.show()


def print_dict(a, keys=None, welcome='', style_sep=': ', end='\n', file=None, print_screen=True, append=True):
    express = welcome
    if keys is None:
        for n in a:
            express += n + style_sep + str(a[n]) + end
    else:
        if type(keys) is str:
            express += keys.capitalize() + style_sep + str(a[keys])
        else:
            for n in keys:
                express += n.capitalize() + style_sep + str(a[n])
                if n is not keys[-1]:
                    express += end
    fprint(express, file, print_screen, append)
    return express


def print_progress_bar(n_current, n_total, message=''):
    x1 = math.floor(n_current / n_total * 10)
    x2 = math.floor(n_current / n_total * 100) % 10
    if x1 == 10:
        message += '\t' + chr(9646) * x1
    else:
        message += '\t' + chr(9646) * x1 + str(x2) + chr(9647) * (9 - x1)
    print('\r'+message, end='')
    time.sleep(0.01)


def print_mat(mat):
    if type(mat) is tc.Tensor:
        mat = mat.numpy()
    for x in mat:
        print(list(x))


def replace_value(x, value0, value_new):
    x_ = np.array(x)
    x_[x_ == value0] = value_new
    return list(x_)


def save(path, file, data, names):
    mkdir(path)
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    tc.save(tmp, os.path.join(path, file))


def search_file(path, exp):
    import re
    content = os.listdir(path)
    exp = re.compile(exp)
    result = list()
    for x in content:
        if re.match(exp, x):
            result.append(os.path.join(path, x))
    return result


def sort_list(a, order):
    return [a[i] for i in order]


# -------------------------------------
# From ZZS
def compare_iterables(a_list, b_list):
    from collections.abc import Iterable
    if isinstance(a_list, Iterable) and isinstance(b_list, Iterable):
        xx = [x for x in a_list if x in b_list]
        if len(xx) > 0:
            return True
        else:
            return False
    else:
        return False


def inverse_permutation(perm):
    # perm is a torch tensor
    if not isinstance(perm, tc.Tensor):
        perm = tc.tensor(perm)
    inv = tc.empty_like(perm)
    inv[perm] = tc.arange(perm.size(0), device=perm.device)
    return inv.tolist()


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    path_flag = os.path.exists(path)
    if not path_flag:
        os.makedirs(path)
    return path_flag


def save(path, file, data, names):
    mkdir(path)
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    tc.save(tmp, os.path.join(path, file))


def load(path_file, names=None, device='cpu'):
    if os.path.isfile(path_file):
        if names is None:
            data = tc.load(path_file)
            return data
        else:
            tmp = tc.load(path_file, map_location=device)
            if type(names) is str:
                data = tmp[names]
                return data
            elif (type(names) is list) or (type(names) is tuple):
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                return tuple(data)
    else:
        return False


def project_path(project='T-Nalg/'):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    return cur_path[:cur_path.find(project) + len(project)]


def output_txt(x, filename='data.txt'):
    np.savetxt(filename, x)


def print_dict(a, keys=None, welcome='', style_sep=': ', end='\n', file=None):
    express = welcome
    if keys is None:
        for n in a:
            express += n + style_sep + str(a[n]) + end
    else:
        if type(keys) is str:
            express += keys.capitalize() + style_sep + str(a[keys])
        else:
            for n in keys:
                express += n.capitalize() + style_sep + str(a[n])
                if n is not keys[-1]:
                    express += end
    if file is None:
        print(express)
    else:
        fprint(express, file)
    return express


def combine_dicts(dic1, dic2):
    # dic1中的重复key值将被dic2覆盖
    return dict(dic1, **dic2)


def random_dates(num, start_date, end_date, if_weekday=True, dates=None, exclude_dates=None):
    # start_date = (year, month, day)
    start = time.mktime(start_date + (0, 0, 0, 0, 0, 0))
    end = time.mktime(end_date + (0, 0, 0, 0, 0, 0))
    if dates is None:
        dates = list()
    if exclude_dates is None:
        exclude_dates = list()
    it_max = 0
    while (len(dates) < num) and (it_max < num * 40):
        t = random.randint(start, end)
        date_touple = time.localtime(t)
        date_touple = time.strftime("%Y%m%d", date_touple)
        # print(date_touple)
        # print(datetime.strptime(date_touple, "%Y%m%d").weekday())
        if (datetime.strptime(date_touple, "%Y%m%d").weekday() < 5) or (
                not if_weekday):
            if (date_touple not in dates) and (date_touple not in exclude_dates):
                dates.insert(0, date_touple)
            # date = list(set(date))
        it_max += 1
    return dates


def fprint(content, file=None, print_screen=True, append=True):
    if file is None:
        file = './record.log'
    if append:
        way = 'ab'
    else:
        way = 'wb'
    with open(file, way, buffering=0) as log:
        log.write((content + '\n').encode(encoding='utf-8'))
    if print_screen:
        print(content)


def warning(string, if_trace_stack=False):
    cprint(string, 'magenta')
    if if_trace_stack:
        trace_stack(3)


def trace_stack(level0=2):
    # print the line and file name where this function is used
    info = stack()
    ns = info.__len__()
    for ns in range(level0, ns):
        cprint('in ' + str(info[ns][1]) + ' at line ' + str(info[ns][2]), 'green')


def now(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d %H:%M:%S'
    return time.strftime(fmt, time.localtime(time.time()))


def kron(mat1, mat2):
    return tc.einsum('ab,cd->acbd', mat1, mat2).reshape(
        mat1.shape[0] * mat2.shape[0], mat1.shape[1] * mat2.shape[1])


def ten_perm(x, pos, pos_first=False):
    perm = list(_ for _ in range(len(x.shape)))
    pos_dim = 1
    shape = list(x.shape)
    for _ in pos:
        perm.remove(_)
        d_ = x.shape[_]
        pos_dim *= d_
        shape.remove(d_)
    if pos_first:
        perm = pos + perm
    else:
        perm = perm + pos
    return x.permute(perm), pos_dim, shape

def tmul(x, y, pos_x=[], pos_y=[]):
    x_new, mul_dim_x, shape_x = ten_perm(x, pos_x)
    y_new, mul_dim_y, shape_y = ten_perm(y, pos_y, pos_first=True)

    shape = shape_x + shape_y
    result = x_new.reshape(-1, mul_dim_x).mm(y_new.reshape(mul_dim_y, -1))
    result = result.reshape(shape)
    return result

def pad_and_cat(tensors, dim=0):
    """
    Catch several tensors at a given dimension, padding small dimensions at the same time.
    """
    # 找到拼接维度以外的每个维度的最大大小
    max_sizes = list(tensors[0].shape)
    for tensor in tensors[1:]:
        for i in range(len(max_sizes)):
            if i != dim:
                max_sizes[i] = max(max_sizes[i], tensor.size(i))
    
    # 对每个张量进行填充
    padded_tensors = []
    for tensor in tensors:
        padding = []
        for i in range(len(max_sizes) - 1, -1, -1):
            if i == dim:
                padding.extend((0, 0))
            else:
                padding.extend((0, max_sizes[i] - tensor.size(i)))
        padded_tensor = tc.nn.functional.pad(tensor, padding)
        padded_tensors.append(padded_tensor)

    # 沿指定维度拼接张量
    result = tc.cat(padded_tensors, dim=dim)
    return result


if __name__ == '__main__':
    x = tc.tensor([[1, 0], [0, 1]])
    y = tc.tensor([[2, 10], [4, 5]])
    z = tmul(x.conj(), y, [0, 1], [0, 1])
    print(z)