from WorkFlow.main import *
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Argument Parser of Init States')
    parser.add_argument('--model_name', type=str, default='PXP')
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--data_type', type=str, default='product')
    parser.add_argument('--loss', type=str, default='multi_mags')
    parser.add_argument('--time_interval', type=float, default=0.02)
    parser.add_argument('--evol_num', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.02)
    # parser.add_argument('--sample_num', type=int, default=10)
    args = parser.parse_args()

    for sample_num in range(1, 21):
        m = 0
        tot_gate_fidelity = 0
        tot_spectrum_diff = 0
        tot_similarity = 0
        for i in range(5):
            m += 1
            init_train_para, init_test_para, return_model_para, evol_para, nn_para, save_para = pack_params(sample_num=sample_num, device=tc.device('cuda:0'), model_para=dict(), csv_file_path='/data/home/scv7454/run/GraduationProject/Data/PXP_multi_sample.csv', **vars(args))
            gate_fidelity, spectrum_diff, similarity = run_with_param(init_train_para, init_test_para, return_model_para, evol_para, nn_para, save_para)
            # print(gate_fidelity, spectrum_diff, similarity)
            tot_gate_fidelity += gate_fidelity
            tot_spectrum_diff += spectrum_diff
            tot_similarity += similarity
        avg_gate_fidelity = tot_gate_fidelity / m
        avg_spectrum_diff = tot_spectrum_diff / m
        avg_similarity = tot_similarity / m
        print('sample_num={}, average_gate_fidelity={}, avgerage_spectrum_diff={}, average_similarity={}'.format(sample_num, float(avg_gate_fidelity), float(avg_spectrum_diff), float(avg_similarity)))
        if avg_gate_fidelity >= 0.98:
            break
