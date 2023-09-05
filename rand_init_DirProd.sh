#!/bin/bash
 
#加载环境，此处加载anaconda环境以及通过anaconda创建的名为myname的环境
module load anaconda/2020.11
module load cuda/11.3
source activate myname

python /data/home/scv7454/run/GraduationProject/rand_init_DirProd.py --train_num 10 --folder rand_dir_prod_init/
python /data/home/scv7454/run/GraduationProject/rand_pred.py --train_num 10 --folder rand_dir_prod_init/
python /data/home/scv7454/run/GraduationProject/rand_init_DirProd.py --train_num 100 --folder rand_dir_prod_init/
python /data/home/scv7454/run/GraduationProject/rand_pred.py --train_num 100 --folder rand_dir_prod_init/
python /data/home/scv7454/run/GraduationProject/rand_init_DirProd.py --train_num 200 --folder rand_dir_prod_init/
python /data/home/scv7454/run/GraduationProject/rand_pred.py --train_num 200 --folder rand_dir_prod_init/
python /data/home/scv7454/run/GraduationProject/rand_init_DirProd.py --train_num 400 --folder rand_dir_prod_init/
python /data/home/scv7454/run/GraduationProject/rand_pred.py --train_num 400 --folder rand_dir_prod_init/
python /data/home/scv7454/run/GraduationProject/rand_init_DirProd.py --train_num 500 --folder rand_dir_prod_init/
python /data/home/scv7454/run/GraduationProject/rand_pred.py --train_num 500 --folder rand_dir_prod_init/
python /data/home/scv7454/run/GraduationProject/rand_init_DirProd.py --train_num 1000 --folder rand_dir_prod_init/
python /data/home/scv7454/run/GraduationProject/rand_pred.py --train_num 1000 --folder rand_dir_prod_init/