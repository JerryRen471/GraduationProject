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