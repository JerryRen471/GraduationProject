import os
import pandas as pd
import re

df_dict = {
    'time_interval': [],
    'train_set_type': [],
    'converge_number': [],
    'converge_gate_fidelity': [],
    'converge_similarity': [],
    'converge_spectrum_var': []
}
job_id = 0
t_list = [0.02 * i for i in range(1, 11)]
data_list = ["product", "non_product", "Z2"]
for train_set_type in data_list:
    for time_interval in t_list:
        job_id += 1
        if job_id >= 27:
            break
        file_path = '/data/home/scv7454/run/convergence_sample_num/{}.out'.format(job_id)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        # 使用正则表达式提取信息
        data = lines[-1]
        try:
            pattern = r"sample_num=(\d+), average_gate_fidelity=([\d.]+), avgerage_spectrum_diff=([\d.]+), average_similarity=([\d.]+)"
            matches = re.search(pattern, data)

            # 提取的信息
            sample_num = int(matches.group(1))
            average_gate_fidelity = float(matches.group(2))
            average_spectrum_diff = float(matches.group(3))
            average_similarity = float(matches.group(4))

            df_dict['time_interval'].append(time_interval)
            df_dict['train_set_type'].append(train_set_type)
            df_dict['converge_number'].append(sample_num)
            df_dict['converge_gate_fidelity'].append(average_gate_fidelity)
            df_dict['converge_similarity'].append(average_similarity)
            df_dict['converge_spectrum_var'].append(average_spectrum_diff)
        except Exception as e:
            print(e)
            print('fix:', train_set_type, time_interval)

df = pd.DataFrame(df_dict)

import matplotlib.pyplot as plt
x1 = df[df['train_set_type']=='product']['time_interval'].unique()
y1 = df[df['train_set_type']=='product']['converge_number'].values
plt.plot(x1, y1)
plt.xticks(x1)
plt.xlabel('time_interval')
plt.ylabel('converge_number')
plt.savefig('/data/home/scv7454/run/GraduationProject/pics/convergence_sample_num_product.png')
plt.close()
x2 = df[df['train_set_type']=='non_product']['time_interval'].unique()
y2 = df[df['train_set_type']=='non_product']['converge_number'].values
plt.plot(x2, y2)
plt.xticks(x2)
plt.xlabel('time_interval')
plt.ylabel('converge_number')
plt.savefig('/data/home/scv7454/run/GraduationProject/pics/convergence_sample_num_non_product.png')
plt.close()

# df.to_csv('/data/home/scv7454/run/convergence_sample_num.csv', index=False)
