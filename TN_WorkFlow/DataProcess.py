import os
import torch as tc
import numpy as np
import scipy.linalg

from matplotlib import pyplot as plt
import pandas as pd
import Library.TensorNetwork as TN

def merge_TN_pack(TN_pack_list):
    '''
    The TN_pack in TN_pack_list must have the same structure, including length, chi, center, device, dtype, etc.
    '''
    new_node_list = deepcopy(TN_pack_list[0].node_list)
    for i in range(len(TN_pack_list) - 1):
        assert TN_pack_list[i].length == TN_pack_list[i+1].length
        assert TN_pack_list[i].chi == TN_pack_list[i+1].chi
        assert TN_pack_list[i].center == TN_pack_list[i+1].center
        assert TN_pack_list[i].device == TN_pack_list[i+1].device
        assert TN_pack_list[i].dtype == TN_pack_list[i+1].dtype
        for j, node in enumerate(TN_pack_list[i+1].node_list):
            new_node_list[j] = tc.cat([new_node_list[j], node], dim=0)
    merged_TN = TN.TensorTrain_pack(tensor_packs=new_node_list, length=TN_pack_list[0].length, chi=TN_pack_list[0].chi, center=TN_pack_list[0].center, device=TN_pack_list[0].device, dtype=TN_pack_list[0].dtype, initialize=False)
    return merged_TN

def cal_gate_fidelity(E:tc.Tensor, U:tc.Tensor):
    n = E.shape[0]
    trace = tc.einsum('aa', U.T.conj() @ E)
    gate_fidelity = 1/(n*(n+1))*(n + tc.abs(trace)**2)
    return gate_fidelity

def cal_similarity(E:tc.Tensor, U:tc.Tensor):
    '''
    E: circuit
    U: real process
    '''
    a = tc.norm(E - U)
    b = 2 * tc.norm(U)
    s = 1 - a/b
    return s

def normalize_pi(n):
    return n - tc.div(n + tc.pi, 2*tc.pi, rounding_mode='trunc') * 2*tc.pi

def cal_spectrum(mat:tc.Tensor):
    energy = tc.log(tc.linalg.eigvals(mat))/1.j
    energy = energy.real
    energy = normalize_pi(energy)
    energy, ind = tc.sort(energy)
    return energy

def write_to_csv(data, csv_file_path, subset):
    """
    向CSV文件写入数据，可以指定接受的数据所对应的列。

    参数:
    data (dict): 要写入的数据字典，其中键为列名，值为对应的数据。
    csv_file_path (str): CSV文件的路径。
    """
    # 将数据转换为 DataFrame
    new_df = pd.DataFrame(data)

    # 检查文件是否存在
    if os.path.exists(csv_file_path):
        # 加载现有的 CSV 数据
        existing_data = pd.read_csv(csv_file_path)

        # 将新数据与现有数据合并
        combined_data = pd.concat([existing_data, new_df], ignore_index=True)
        combined_data = combined_data.sort_values(subset)

        # 去重，保留最后出现的行
        # combined_data = combined_data.drop_duplicates()
        #     subset=subset, keep='last'
        # )
    else:
        # 文件不存在，直接使用新数据  
        combined_data = new_df
    
    # 保存更新后的数据到 CSV 文件
    combined_data.to_csv(csv_file_path, index=False, mode='w')

def main(
    qc_mat:tc.Tensor, 
    evol_mat:tc.Tensor, 
    results:dict,
    pic_path:str,
    csv_file_path:str,
    **para):

    # 对所有不同损失函数训练得到的数据训练过程中的保真度的变化
    train_fide = results['train_fide']
    test_fide = results['test_fide']

    train_loss = results['train_loss']
    test_loss = results['test_loss']
    x = list(range(0, len(train_loss)))

    legends = []
    plt.plot(x, train_loss, label='train loss')
    plt.plot(x, test_loss, label= 'test loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(pic_path+'/loss_num{:d}.svg'.format(para['evol_num']))
    plt.close()

    legends = []
    plt.plot(x, train_fide, label='train fidelity')
    plt.plot(x, test_fide, label= 'test fidelity')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('fidelity')
    plt.savefig(pic_path+'/fidelity_num{:d}.svg'.format(para['evol_num']))
    plt.close()

    data = dict(para)
    data['train_loss'] = [float(train_loss[-1])]
    data['test_loss'] = [float(test_loss[-1])]
    data['train_fide'] = [float(train_fide[-1])]
    data['test_fide'] = [float(test_fide[-1])]

    return_list = []
    gate_fidelity = cal_gate_fidelity(qc_mat, evol_mat)
    data['gate_fidelity'] = [float(gate_fidelity)]
    return_list.append(gate_fidelity)

    similarity = cal_similarity(qc_mat, evol_mat)
    data['similarity'] = [float(similarity)]
    return_list.append(similarity)

    if para['time_interval'] != 0:
        qc_energy = cal_spectrum(qc_mat) / para['time_interval']
        evol_energy = cal_spectrum(evol_mat) / para['time_interval']
        spectrum_diff = tc.var(qc_energy - evol_energy)
        data['spectrum_diff'] = [float(spectrum_diff)]
        return_list.append(spectrum_diff)

        qc_mat_np = qc_mat.numpy()
        evol_mat_np = evol_mat.numpy()
        H_qc = scipy.linalg.logm(qc_mat_np)/ 1.j / para['time_interval']
        H_evol = scipy.linalg.logm(evol_mat_np)/ 1.j / para['time_interval']
    
        plt.imshow(np.abs(H_qc), cmap='Greys', interpolation='nearest', vmin=np.min(np.abs(H_qc)), vmax=np.max(np.abs(H_evol)))
        plt.colorbar(label='Absolute Value')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.title('Heatmap of H_qc')
        plt.savefig(pic_path+'/H_qc_heatmap_num{:d}.svg'.format(para['evol_num']))
        plt.close()

        plt.imshow(np.abs(H_evol), cmap='Greys', interpolation='nearest', vmin=np.min(np.abs(H_qc)), vmax=np.max(np.abs(H_evol)))
        plt.colorbar(label='Absolute Value')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.title('Heatmap of H_evol')
        plt.savefig(pic_path+'/H_evol_heatmap_num{:d}.svg'.format(para['evol_num']))
        plt.close()

        abs_diff = np.abs(H_qc - H_evol)
        plt.imshow(abs_diff, cmap='Greys', interpolation='nearest')
        plt.colorbar(label='Absolute Difference Value')
        plt.xlabel('Column Index')
        plt.ylabel('Row Index')
        plt.title('Heatmap of Absolute Difference')
        plt.savefig(pic_path+'/abs_diff_heatmap_num{:d}.svg'.format(para['evol_num']))
        plt.close()

        H_diff = np.mean(abs_diff)
        data['H_diff'] = [float(H_diff)]
        return_list.append(H_diff)
    
    write_to_csv(data, csv_file_path, subset=list(para.keys()))
    return tuple(return_list)