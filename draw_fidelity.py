from matplotlib import pyplot as plt
import re

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--folder', type=str, default='/log_loss_multi_mags/dn')
args = parser.parse_args()

data_path = 'GraduationProject/Data'+args.folder
pic_path = 'GraduationProject/pics'+args.folder

train_num_pattern = r"train_num=[\d]+"
train_loss_pattern = r"train_loss:[\S]+"
test_loss_pattern = r"test_loss:[\S]+"
train_fide_pattern = r"train_fide:[\S]+"
test_fide_pattern = r"test_fide:[\S]+"

train_num_list = list()
train_loss = list()
test_loss = list()
train_fide = list()
test_fide = list()

def add_to_list(pattern, list_, line):
    ret = re.match(pattern, line)
    if ret != None:
        num = float(re.match(r"[+,-,0-9,e,.]+", ret.group()))
        list_.append(num)

with open(data_path+'/fin_loss_train_num.txt', 'r') as f:
    for line in f.readlines():
        add_to_list(train_num_pattern, train_num_list, line)
        add_to_list(train_loss_pattern, train_loss, line)
        add_to_list(train_fide_pattern, train_fide, line)
        add_to_list(test_loss_pattern, test_loss, line)
        add_to_list(test_fide_pattern, test_fide, line)

x = train_num_list

legends = []
plt.plot(x, train_loss, label='train loss')
plt.plot(x, test_loss, label= 'test loss')
plt.legend()
plt.xlabel('train_num')
plt.ylabel('loss')
plt.savefig(pic_path+'/diff_train_num_loss.svg')
plt.close()

legends = []
plt.plot(x, train_fide, label='train fidelity')
plt.plot(x, test_fide, label= 'test fidelity')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('fidelity')
plt.savefig(pic_path+'/diff_train_num_fide.svg')
plt.close()