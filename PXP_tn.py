from TN_WorkFlow.main import *
import argparse

if __name__ == "__main__":

    # 创建一个新的解析器
    parser = argparse.ArgumentParser(description='Combined Argument Parser')

    # 创建物理模型参数组
    model_group = parser.add_argument_group('Physical Model Parameters')

    # 创建训练参数组
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--sample_num', type=int, default=10, help='Number of samples')
    train_group.add_argument('--length', type=int, default=10, help='Length of the input')
    train_group.add_argument('--data_type', type=str, default='product', help='Type of data')
    train_group.add_argument('--loss', type=str, default='multi_mags', help='Loss function type')
    train_group.add_argument('--time_interval', type=float, default=0.02, help='Time interval for training')
    train_group.add_argument('--evol_num', type=int, default=10, help='Number of evolutions')
    train_group.add_argument('--tau', type=float, default=0.02, help='Time constant')

    # 解析参数
    args = parser.parse_args()

    # 将参数分组
    model_params = {
    }

    train_params = {
        'sample_num': args.sample_num,
        'length': args.length,
        'data_type': args.data_type,
        'loss': args.loss,
        'time_interval': args.time_interval,
        'evol_num': args.evol_num,
        'tau': args.tau
    }

    chi = 10
    m = 0
    for i in range(2):
        m += 1
        init_train_para, init_test_para, return_model_para, evol_para, nn_para, save_para = pack_params(chi=chi, model_name='PXP', model_para=model_params, device=tc.device('cuda:0'), csv_file_path='/data/home/scv7454/run/GraduationProject/Data/PXP_tn.csv', **train_params)
        return_dict = run_with_param(init_train_para, init_test_para, return_model_para, evol_para, nn_para, save_para)
        for key, value in return_dict.items():
            print(key, value, end="\t")
            print('')
