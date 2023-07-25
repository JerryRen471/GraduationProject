#!/bin/bash
 
#加载环境，此处加载anaconda环境以及通过anaconda创建的名为myname的环境
module load anaconda/2020.11
module load cuda/11.3
source activate myname
 
#python程序运行，需在.py文件指定调用GPU，并设置合适的线程数，batch_size大小等
# python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 100 --time 100
# python /data/home/scv7454/run/GraduationProject/ADQC_tot.py --pt_it 100 --time 100
python /data/home/scv7454/run/GraduationProject/draw_tot.py --pt_it=10 --time=1000 -s=n