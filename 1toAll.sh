#!/bin/bash
 
#加载环境，此处加载anaconda环境以及通过anaconda创建的名为myname的环境
module load anaconda/2020.11
module load cuda/11.3
source activate myname

python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 10 --time 100 --hl 2
python /data/home/scv7454/run/GraduationProject/ADQC.py --pt_it 10 --time 100
python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 5 --time 100 --hl 2
python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 15 --time 100 --hl 2
python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 20 --time 100 --hl 2
python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 25 --time 100 --hl 2
python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 50 --time 100 --hl 2
python /data/home/scv7454/run/GraduationProject/TimeEvolution.py --pt_it 100 --time 100 --hl 2
python /data/home/scv7454/run/GraduationProject/1toAll.py --pt_it 5 --time 100
python /data/home/scv7454/run/GraduationProject/1toAll.py --pt_it 10 --time 100
python /data/home/scv7454/run/GraduationProject/1toAll.py --pt_it 15 --time 100
python /data/home/scv7454/run/GraduationProject/1toAll.py --pt_it 20 --time 100
python /data/home/scv7454/run/GraduationProject/1toAll.py --pt_it 25 --time 100
python /data/home/scv7454/run/GraduationProject/1toAll.py --pt_it 50 --time 100
python /data/home/scv7454/run/GraduationProject/1toAll.py --pt_it 100 --time 100