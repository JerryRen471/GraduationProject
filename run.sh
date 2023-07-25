#!/bin/bash
 
#加载环境，此处加载anaconda环境以及通过anaconda创建的名为myname的环境
module load anaconda/2020.11
module load cuda/11.3
source activate myname
 
#python程序运行，需在.py文件指定调用GPU，并设置合适的线程数，batch_size大小等
# python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 5
python /data/home/scv7454/run/GraduationProject/ADQC.py
# python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 10
# python /data/home/scv7454/run/GraduationProject/ADQC.py --pt_it 10
# python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 15
# python /data/home/scv7454/run/GraduationProject/ADQC.py --pt_it 15
# python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 20
# python /data/home/scv7454/run/GraduationProject/ADQC.py --pt_it 20
# python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 25
# python /data/home/scv7454/run/GraduationProject/ADQC.py --pt_it 25
# python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 50
# python /data/home/scv7454/run/GraduationProject/ADQC.py --pt_it 50
# python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 100
# python /data/home/scv7454/run/GraduationProject/ADQC.py --pt_it 100
# python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 1000
# python /data/home/scv7454/run/GraduationProject/ADQC.py --pt_it 1000
