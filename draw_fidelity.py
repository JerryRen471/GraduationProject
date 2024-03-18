from cProfile import label
from matplotlib import pyplot as plt
import re

import argparse

import numpy as np
parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--folder', type=str, default='/loss_multi_mags/dn')
args = parser.parse_args()

data_path = 'GraduationProject/Data'+args.folder
pic_path = 'GraduationProject/pics'+args.folder

train_num_pattern = r"train_num=[\d]+"
train_loss_pattern = r"train_loss:[\S]+"
test_loss_pattern = r"test_loss:[\S]+"
train_fide_pattern = r"train_fide:[\S]+"
test_fide_pattern = r"test_fide:[\S]+"

train_num_list = [[]]
train_loss_list = [[]]
test_loss_list = [[]]
train_fide_list = [[]]
test_fide_list = [[]]

def pattern_num(pattern, line):
    ret = re.search(pattern, line)
    if ret != None:
        num = float(re.search("[+\-e\.0-9]+$", ret.group()).group())
        return num
    else:
        return None

def add_to_list(pattern, list, line):
    ret = re.search(pattern, line)
    if ret != None:
        num = float(re.search("[+\-e\.0-9]+$", ret.group()).group())
        list.append(num)

train_num = None
with open(data_path+'/fin_loss_train_num.txt', 'r') as f:
    for line in f.readlines():
        if line == '---\n':
            train_num_list.append([])
            train_loss_list.append([])
            test_loss_list.append([])
            train_fide_list.append([])
            test_fide_list.append([])
        else:
            train_num = pattern_num(train_num_pattern, line)    
            add_to_list(train_num_pattern, train_num_list[-1], line)
            add_to_list(train_loss_pattern, train_loss_list[-1], line)
            add_to_list(train_fide_pattern, train_fide_list[-1], line)
            add_to_list(test_loss_pattern, test_loss_list[-1], line)
            add_to_list(test_fide_pattern, test_fide_list[-1], line)

# with open(data_path+'/fin_loss_train_num.txt', 'a') as f:
#     f.write('---\n')

# x = train_num_list

legends = []
for i in range(len(train_num_list)):
    plt.plot(train_num_list[i], train_loss_list[i], marker='+', label='train loss')
    plt.plot(train_num_list[i], test_loss_list[i], marker='x', label= 'test loss')
    plt.legend()
    plt.xlabel('train_num')
    plt.ylabel('loss')
    plt.savefig(pic_path+'/{:d}diff_train_num_loss.svg'.format(i))
    plt.close()

legends = []
for i in range(len(train_num_list)):
    plt.plot(train_num_list[i], train_fide_list[i], marker='+', label='train fidelity')
    plt.plot(train_num_list[i], test_fide_list[i], marker='x', label= 'test fidelity')
    plt.legend()
    plt.xlabel('train_num')
    plt.ylabel('fidelity')
    plt.savefig(pic_path+'/{:d}diff_train_num_fide.svg'.format(i))
    plt.close()

# plot gate_fidelity between adqc and evolution matrix

def draw(data_path, pic_path, label, picname):
    gate_fidelity_list = [[]]
    train_num_list = [[]]
    with open(data_path, 'r') as f:
        for line in f.readlines():
            if line == '---\n':
                train_num_list.append([])
                gate_fidelity_list.append([])
            else:
                sim = float(line.split('\t')[0])
                train_num = int(line.split('\t')[1])
                gate_fidelity_list[-1].append(sim)
                train_num_list[-1].append(train_num)

    legend = []
    for i in range(len(train_num_list)):
        plt.plot(train_num_list[i], gate_fidelity_list[i], marker='+')
# plt.plot(x, test_fide, marker='x', label= 'test fidelity')
        plt.legend()
        plt.xlabel('train_num')
        plt.ylabel(label)
        plt.savefig(pic_path+'/{:d}'.format(i)+picname)
        plt.close()

draw(data_path+'/gate_fidelity.txt', pic_path, label='gate_fidelity', picname='diff_train_num_gate_fide.svg')
draw(data_path+'/similarity.txt', pic_path, label='similarity', picname='similarity_diff_train.svg')
