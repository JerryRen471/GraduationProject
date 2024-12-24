from WorkFlow.main import *
import argparse

if __name__ == "__main__":

    # 创建一个新的解析器
    parser = argparse.ArgumentParser(description='Combined Argument Parser')

    # 创建随机线路参数组
    model_group = parser.add_argument_group('Random Circuit Parameters')
    model_group.add_argument('--depth', type=int, default=4)

    # 创建训练参数组
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--sample_num', type=int, default=10, help='Number of samples')
    train_group.add_argument('--length', type=int, default=10, help='Length of the input')
    train_group.add_argument('--data_type', type=str, default='product', help='Type of data')
    train_group.add_argument('--loss', type=str, default='multi_mags', help='Loss function type')

    # 解析参数
    args = parser.parse_args()

    # 将参数分组
    model_params = {
        'depth': args.depth,
        'length': args.length
    }

    train_params = {
        'sample_num': args.sample_num,
        'length': args.length,
        # 'evol_num': 1,
        'data_type': args.data_type,
        'loss': args.loss,
    }

    m = 0
    tot_gate_fidelity = 0
    tot_spectrum_diff = 0
    tot_similarity = 0
    for i in range(5):
        m += 1
        init_train_para, init_test_para, return_model_para, evol_para, nn_para, save_para = pack_rc_params(model_name='random_circuit', model_para=model_params, device=tc.device('cuda:0'), csv_file_path='/data/home/scv7454/run/GraduationProject/Data/random_circuit.csv', **train_params)
        gate_fidelity, similarity = run_with_param(init_train_para, init_test_para, return_model_para, evol_para, nn_para, save_para)
        print(gate_fidelity, similarity)
        tot_gate_fidelity += gate_fidelity
        tot_similarity += similarity
    avg_gate_fidelity = tot_gate_fidelity / m
    avg_similarity = tot_similarity / m
    print('average_gate_fidelity={}, average_similarity={}'.format(float(avg_gate_fidelity), float(avg_similarity)))

