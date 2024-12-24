import os
import numpy as np
import re
import pandas as pd
import traceback
# from draw_rand_PXP import write_to_csv

def write_to_csv(data, csv_file_path, subset, dtypes):
    """
    向CSV文件写入数据，可以指定接受的数据所对应的列。

    参数:
    data (dict): 要写入的数据字典，其中键为列名，值为对应的数据。
    csv_file_path (str): CSV文件的路径。
    """
    # 将数据转换为 DataFrame
    new_df = pd.DataFrame(data).astype(dtypes)

    # 检查文件是否存在
    if os.path.exists(csv_file_path):
        # 加载现有的 CSV 数据
        existing_data = pd.read_csv(csv_file_path).astype(dtypes)

        # 将新数据与现有数据合并
        combined_data = pd.concat([existing_data, new_df], ignore_index=True)
        combined_data = combined_data.sort_values(subset)

        # 去重，保留最后出现的行
        combined_data = combined_data.drop_duplicates(keep='last')
    else:
        # 文件不存在，直接使用新数据  
        combined_data = new_df
    
    # 保存更新后的数据到 CSV 文件
    combined_data.to_csv(csv_file_path, index=False)

def find_error_files(directory):
    error_files = []  # 用于记录错误文件名
    for filename in os.listdir(directory):
        if filename.endswith('.err'):
            with open(os.path.join(directory, filename), 'r') as file:
                content = file.read()
                if 'FileNotFoundError:' in content:
                    error_files.append(int(filename[:-4]))  # 记录文件名
    return error_files

def delete_files(directory, n):
    """delete the output files with number smaller than n"""
    for filename in os.listdir(directory):
        if filename.endswith('.out') or filename.endswith('.err'):
            try:
                file_number = int(filename.split('.')[0])
                if file_number < n:
                    file_path = os.path.join(directory, filename)
                    os.remove(file_path)
            except Exception as e:
                print(e)

def format_content(content):
    formatted_dict = {}
    for line in content:
        key, value = line.split('=')
        formatted_dict[key.strip()] = value.strip()
    return formatted_dict

def extract_content(file_path):
    # file_path = '/data/home/scv7454/run/' + str(n) + '.out'
    content = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        model_name = lines[0].strip()
        for line in lines[1:]:
            if line.startswith('evol_mat.shape'):
                break
            line = line.strip()
            if line:
                content.append(line)
    formatted_content = format_content(content)
    return model_name, formatted_content

def extract_info(file_path):
    # 从文件中读取内容
    # file_path = '/data/home/scv7454/run/1340756.out'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 使用正则表达式提取包含信息的行
    pattern = r"The last line of the Python file output is: (.*)"
    info_line = None
    for line in lines:
        match = re.search(pattern, line)
        if match:
            info_line = match.group(1)
            break

    # 提取具体的数据信息
    data_info = {}
    if info_line:
        pattern = r"\(([a-zA-Z_]+?)=\[\'?(.+?)\'?\]\)"
        matches = re.findall(pattern, info_line)
        for match in matches:
            key = match[0]
            value = match[1]
            data_info[key] = [value]

    # 返回提取的数据信息
    return data_info

def extract_sub_jobs(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        sub_jobs = []
        for line in lines:
            if line.startswith('job_id'):
                # 使用正则表达式提取job_id的值
                match = re.search(r'job_id:\s*(\d+)', line)
                if match:
                    sub_job = match.group(1)
                    sub_jobs.append(int(sub_job))
    return sub_jobs

def extract_info_to_csv(directory, csv_path, model_name, job_list, subset, dtypes):
    for filename in os.listdir(directory):
        if filename.endswith('.out'):
            try:
                file_number = int(filename.split('.')[0].lstrip('PXP'))
            except Exception as e:
                # print(e)
                file_number = 0
            try:
                if file_number in job_list:
                    file_path = os.path.join(directory, filename)
                    read_model_name, content = extract_content(file_path)
                    # 过滤掉不是对应的model的文件
                    # train_set_type = content.get('data_type', []) 
                    if read_model_name == model_name:
                        data_info = extract_info(file_path)
                        # write_to_csv(data_info, '/data/home/scv7454/run/GraduationProject/Data/xorX.csv', subset, dtypes)
                        if len(data_info) < (len(dtypes)):
                            with open('GraduationProject/fix_{}.txt'.format(model_name), 'a') as file:
                                line = ' '.join(content.values()) + '\n'
                                file.write(line)
                        else:
                            write_to_csv(data_info, csv_path, subset, dtypes)
                    # elif train_set_type == []:
                    #     content = extract_content(file_number)
                    #     with open('GraduationProject/fix_xorX.txt', 'a') as file:
                    #         line = ' '.join(content.values()) + '\n'
                    #         file.write(line)
                    # print(data_info)
            except Exception as e:
                print(file_number)
                traceback.print_exc()
                line_number = traceback.extract_tb(e.__traceback__)[-1].lineno
                print("Error occurred on line:", line_number)
                # print(e)

# 示例用法
if __name__ == "__main__":
    directory_path = '/data/home/scv7454/run/20241110'  # 替换为你的目录路径
    csv_path = '/data/home/scv7454/run/GraduationProject/Data/Heis.csv'
    model_name = 'xorX'
    # delete_files(directory_path, 2100)
    # errors = find_error_files(directory_path)
    # errors = np.sort(np.array(errors))
    # print(errors)
    # print(len(errors))
    # # print(extract_content(1336377))
    # with open('GraduationProject/fix.txt', 'w') as file:
    #     file.write('')
    # for n in errors:
    #     content = extract_content(n)
    #     with open('GraduationProject/fix.txt', 'a') as file:
    #         line = ' '.join(content.values()) + '\n'
    #         file.write(line)

    # file_path = '/data/home/scv7454/run/1340756.out'
    # data_info = extract_info(file_path)
    # print(data_info)

    dtypes = {
        'length': int,
        # 'delta': float,
        # 'J': float,
        # 'lambda': float,
        'time_interval': float,
        'evol_num': int,
        'sample_num': int,
        'train_set_type': str,
        'entangle_dim': int,
        'loss': str,
        'gate_fidelity': float,
        'spectrum_diff': float,
        'H_diff': float,
        'train_loss': float,
        'test_loss': float,
        'train_fidelity': float,
        'test_fidelity': float
    }
    subset=[
        # 'delta', 'J', 'lambda', 
        'train_set_type', 'loss', 'length', 'time_interval', 'evol_num', 'sample_num', 'entangle_dim'
        ]

    # job_list = extract_sub_jobs('/data/home/scv7454/run/slurm-1382029.out')
    job_list = [i for i in range(301, 361)]

    extract_info_to_csv(directory=directory_path, csv_path=csv_path, model_name=model_name, job_list=job_list, subset=subset, dtypes=dtypes)
