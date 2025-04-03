import sys
sys.path.append('/data/home/scv7454/run/GraduationProject')
from TN_WorkFlow import InitStates, TimeEvol, TrainModel, DataProcess
import argparse
import ast
import os
import re
import torch as tc

def search_qc(folder_path, sample_num, evol_num):
    # 使用正则表达式匹配文件名中的 evol 和 sample
    pattern = re.compile(r'qc_param_sample_(\d+)_evol_(\d+)\.pt')

    max_sample = None
    max_evol = None
    target_file = None

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            match = pattern.search(filename)
            if match:
                temp_sample = int(match.group(1))
                temp_evol = int(match.group(2))

                # 检查 temp_sample 小于给定的 sample_num
                if temp_sample < sample_num:
                    # 如果 max_sample 为空或 temp_sample 大于 max_sample，更新 max_sample 和 max_evol
                    if max_sample is None or temp_sample > max_sample:
                        max_sample = temp_sample
                        max_evol = temp_evol  # 更新 max_evol
                        target_file = filename
                    # 如果 temp_sample 等于 max_sample，检查 temp_evol
                    elif temp_sample == max_sample and temp_evol < evol_num:
                        if max_evol is None or temp_evol > max_evol:
                            max_evol = temp_evol
                            target_file = filename

    return target_file

def run_with_param(init_train_para, init_test_para, model_para, evol_para, nn_para, save_para):
    '''
    init_train_para: 包含初态的参数，有 number, length, type, type_para
    init_test_para: 包含初态的参数，有 number, length, type, type_para
    model_para: 包含物理模型的参数，有 length, model_name以及模型包含的参数
    evol_para: 包含时间演化的参数，有 evol_num, time_interval, tau （对train_set进行演化的参数，test_set的演化参数固定evol_num=1）
    nn_para: 包含神经网络的参数，有 loss_type, it_time, batch_size, print_time, recurrent_time
    save_para: 包含保存的参数，有 csv_file_path ，加上 model_para/init_train_para/evol_para/nn_para
    '''

    # 生成训练集和测试集
    model_name = model_para['model_name']
    model_para = {key: value for key, value in model_para.items() if (key != 'model_name')}

    train_init = InitStates.main(init_train_para)
    evol_train_para = dict()
    # if evol_para['time_interval'] != 0:
    evol_train_para['tau'] = evol_para['tau']
    evol_train_para['print_time'] = evol_para['time_interval']
    evol_train_para['time_tot'] = evol_para['time_interval'] * evol_para['evol_num']
    train_label, evol_mat = TimeEvol.main(model_name=model_name, model_para=model_para, init_states=train_init, evol_para=evol_train_para, return_mat=True)
    train_input = tc.cat((train_init.unsqueeze(1), train_label[:, :-1]), dim=1)

    test_init = InitStates.main(init_test_para)
    evol_test_para = dict()
    # if evol_para['time_interval'] != 0:
    evol_test_para['print_time'] = evol_para['time_interval']
    evol_test_para['time_tot'] = evol_para['time_interval']
    test_label = TimeEvol.main(model_name=model_name, model_para=model_para, init_states=test_init, evol_para=evol_test_para)
    test_input = tc.cat((test_init.unsqueeze(1), test_label[:, :-1]), dim=1)

    data = dict()
    merge = lambda x: x.reshape(x.shape[0]*x.shape[1], *x.shape[2:])
    data['train_set'] = merge(train_input)
    data['train_label'] = merge(train_label)
    data['test_set'] = merge(test_input)
    data['test_label'] = merge(test_label)

    # 加载并训练神经网络
    folder = "/{model_name}/length{length}/loss_{loss}/{time_interval}/{data_type}".format(
        model_name=model_name, length=model_para['length'], loss=nn_para['loss_type'], time_interval=evol_para['time_interval'], data_type=init_train_para['type'])
    data_path = "GraduationProject/Data" + folder
    os.makedirs(data_path, exist_ok=True)
    old_param = None
    old_qc_path = search_qc(folder_path=data_path, evol_num=evol_para['evol_num'], sample_num=init_train_para['number'])
    if old_qc_path != None:
        old_param = tc.load(data_path + '/' + old_qc_path)
        # os.remove(data_path + '/' + old_qc_path)
    qc, results, nn_para = TrainModel.main(qc_type='ADQC', init_param=old_param, data=data, nn_para=nn_para)
    new_qc_path = data_path + '/qc_param_sample_{}_evol_{}.pt'.format(init_train_para['number'], evol_para['evol_num'])
    tc.save(qc.state_dict(), new_qc_path)
    # for key, value in results.items():
    #     results[key] = value.cpu()
    # np.save(path+'/adqc_result_sample_{:d}_evol_{:d}'.format(args.sample_num, args.evol_num), results_adqc)

    pic_path = "GraduationProject/pics" + folder
    os.makedirs(pic_path, exist_ok=True)
    qc.single_state = False
    E = tc.eye(2**nn_para['length_in'], dtype=nn_para['dtype'], device=nn_para['device'])
    shape_ = [E.shape[0]] + [2] * nn_para['length_in']
    E = E.reshape(shape_)
    with tc.no_grad():
        for _ in range(nn_para['recurrent_time']):
            E = qc(E)
        qc_mat = E.reshape([E.shape[0], -1])
    print('qc_mat.shape:', qc_mat.shape)
    print('evol_mat.shape:', evol_mat.shape)
    
    return_tuple = DataProcess.main(qc_mat=qc_mat.cpu(), evol_mat=evol_mat.cpu(), results=results, pic_path=pic_path, **save_para)
    return return_tuple

def pack_params(
        chi:int, 
        model_name:str, 
        model_para:dict, 
        length:int, 
        data_type:str, 
        loss:str, 
        time_interval:float, 
        evol_num:int, 
        tau: float,
        sample_num:int, 
        csv_file_path:str,
        ini_way:str='identity',
        rec_time:int=1, 
        depth:int=4, 
        device=tc.device('cuda:0'), 
        dtype=tc.complex128, 
        **kwargs
        ):
    init_train_para = {
        'chi': chi,
        'type': data_type,
        'length': length, 
        'number': sample_num, 
        'device': device, 
        'dtype': dtype
    }
    init_train_para = dict(init_train_para, **kwargs, **model_para)
    init_test_para = {
        'chi': chi,
        'type': 'product',
        'length': length,
        'number': 100, 
        'device': device, 
        'dtype': dtype
    }

    return_model_para = {
        'model_name': model_name,
        'chi': chi,
        'length': length,
        'device': device,
        'dtype': dtype
    }
    return_model_para = dict(return_model_para, **model_para)

    evol_para = {
        'chi': chi,
        'evol_num': evol_num,
        'time_interval': time_interval,
        'tau': tau,
        'device': device,
        'dtype': dtype
    }

    nn_para = {
        'loss_type': loss,
        'length_in': length,
        'length_out': length,
        'ini_way': ini_way,
        'recurrent_time': rec_time,
        'depth': depth,
        'device': device,
        'dtype': dtype
    }

    save_para = {
        'chi': chi,
        'model_name': model_name,
        'length': length,
        'csv_file_path': csv_file_path,
        'evol_num': evol_num,
        'sample_num': sample_num,
        'time_interval': time_interval,
        'loss_type': loss,
        'data_type': data_type
    }
    save_para = dict(save_para, **model_para)
    
    return init_train_para, init_test_para, return_model_para, evol_para, nn_para, save_para

def pack_rc_params(
        model_name:str, 
        model_para:dict, 
        length:int, 
        data_type:str, 
        loss:str, 
        # evol_num:int, 
        sample_num:int, 
        # time_interval:float,
        csv_file_path:str,
        ini_way:str='identity',
        rec_time:int=1, 
        depth:int=4, 
        device=tc.device('cuda:0'), 
        dtype=tc.complex128, 
        **kwargs
        ):
    init_train_para = {
        'type': data_type,
        'length': length, 
        'number': sample_num, 
        'device': device, 
        'dtype': dtype
    }
    init_train_para = dict(init_train_para, **kwargs)
    init_test_para = {
        'type': 'product',
        'length': length,
        'number': 100, 
        'device': device, 
        'dtype': dtype
    }

    return_model_para = {
        'model_name': model_name,
        'length': length,
        'device': device,
        'dtype': dtype
    }
    return_model_para = dict(return_model_para, **model_para)

    evol_para = {
        'evol_num': 1,
        'time_interval': 0,
        'tau': 0,
        'device': device,
        'dtype': dtype
    }

    nn_para = {
        'loss_type': loss,
        'length_in': length,
        'length_out': length,
        'ini_way': ini_way,
        'recurrent_time': rec_time,
        'depth': depth,
        'device': device,
        'dtype': dtype
    }

    save_para = {
        'model_name': model_name,
        'length': length,
        'csv_file_path': csv_file_path,
        'evol_num': 1,
        'time_interval': 0,
        'sample_num': sample_num,
        'loss_type': loss,
        'data_type': data_type
    }
    save_para = dict(save_para, **model_para)
    
    return init_train_para, init_test_para, return_model_para, evol_para, nn_para, save_para

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Argument Parser of Init States')
    parser.add_argument('--model_name', type=str, default='PXP')
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--data_type', type=str, default='product')
    parser.add_argument('--loss', type=str, default='multi_mags')
    parser.add_argument('--time_interval', type=float, default=0.02)
    parser.add_argument('--evol_num', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.02)
    parser.add_argument('--sample_num', type=int, default=10)
    args = parser.parse_args()

    init_train_para, init_test_para, return_model_para, evol_para, nn_para, save_para = pack_params(device=tc.device('cuda:0'), model_para=dict(), csv_file_path='/data/home/scv7454/run/GraduationProject/Data/PXP_test.csv', **vars(args))
    gate_fidelity, spectrum_diff, similarity = run_with_param(init_train_para, init_test_para, return_model_para, evol_para, nn_para, save_para)
    print(gate_fidelity, spectrum_diff, similarity)