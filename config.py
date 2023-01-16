import random
import numpy as np
import torch
from typing import List
from utils import *


# Fix seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)


# Global valuable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTCOME_DIR = os.path.join(BASE_DIR, 'result')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
COMMON_CONFIG_PATH = os.path.join(CONFIG_DIR, 'common.txt')
INSIDE_CONFIG_PATH = os.path.join(CONFIG_DIR, 'inside_list.txt')
COLUMN_CONFIG_PATH = os.path.join(CONFIG_DIR, 'flow_column_map.txt')
TRAIN_CONFIG_PATH = os.path.join(CONFIG_DIR, 'train_dir.txt')
TEST_CONFIG_PATH = os.path.join(CONFIG_DIR, 'test_dir.txt')


class Config:

    class __Path:

        def __init__(self, outcome_name: str):
            self.outcome_dir = os.path.join(OUTCOME_DIR, outcome_name)
            self.result_dir = os.path.join(self.outcome_dir, 'result')
            self.vgm_path = os.path.join(self.outcome_dir, 'vgm.pkl')
            self.rfre_encoder_path = os.path.join(self.outcome_dir, 'rfre_encder.pkl')
            self.model_path = os.path.join(self.outcome_dir, 'model.pth')
            self.model_txt_path = os.path.join(self.outcome_dir, 'model.txt')
            self.threshold_path = os.path.join(self.outcome_dir, 'threshold.txt')
            self.auroc_path = os.path.join(self.outcome_dir, 'auroc.csv')
            self.eval_path = os.path.join(self.outcome_dir, 'eval.csv')
            self.roc_curve_path = os.path.join(self.outcome_dir, 'roc_curve.png')
            self.scatter_plot_path = os.path.join(self.outcome_dir, 'scatter_plot.png')
            create_directory(self.outcome_dir, recursive=True)
            create_directory(self.result_dir, recursive=True)

    def __init__(self, dataset_name: str, outcome_name: str = format_date(6)):
        self.path = self.__Path(outcome_name)
        self.dataset_name = dataset_name
        self.__train_dir_list = []
        self.__test_dir_list = []
        self.__inside_ip_set = set()
        self.__column_index = {
            'sip': None,
            'sport': None,
            'dip': None,
            'dport': None,
            'time_start': None,
            'time_end': None,
            'duration': None,
            'in_packets': None,
            'out_packets': None,
            'in_bytes': None,
            'out_bytes': None,
            'label': None
        }
        self.profiling_target = 'dip'
        self.time_window = 300
        self.interface = 32
        self.epochs = 250
        self.batch_size = 256

        self.build()
        self.show_info()

    def build(self):
        self.__load_common_variable()
        self.__train_dir_list += self.parse_config_file(TRAIN_CONFIG_PATH, self.dataset_name)
        self.__test_dir_list += self.parse_config_file(TEST_CONFIG_PATH, self.dataset_name)
        self.__load_inside_ip_set()
        self.__load_column_index()

    def show_info(self):
        print('################################################################')
        print(f'# Dataset: {self.dataset_name}')
        print(f'# Outcome directory path: {self.path.outcome_dir}')
        print(f'# Profiling target: {self.profiling_target}')
        print(f'# time window(s): {self.time_window}, interface: /{self.interface}')
        print(f'# epochs: {self.epochs}, batch size: {self.batch_size}')
        print('################################################################\n')

    @property
    def inside_ip_set(self) -> set:
        return self.__inside_ip_set

    @property
    def column_index(self) -> dict:
        return self.__column_index

    @property
    def train_dir_list(self) -> List[str]:
        return self.__train_dir_list

    @property
    def test_dir_list(self) -> List[str]:
        return self.__test_dir_list

    def __load_common_variable(self):
        for line in self.parse_config_file(COLUMN_CONFIG_PATH, 'COMMON'):
            variable, value = tuple(map(str.strip, line.split('=')))
            if variable == 'profiling_target':
                self.profiling_target = value
            elif variable == 'time_window':
                self.time_window = value
            elif variable == 'interface':
                self.interface = value
            elif variable == 'epochs':
                self.epochs = value
            elif variable == 'batch_size':
                self.batch_size = value

    def __load_column_index(self):
        tmp_flow_csv_path = os.path.join(self.train_dir_list[0], os.listdir(self.train_dir_list[0])[0])
        with open(tmp_flow_csv_path, 'r') as f:
            split_column_line = list(map(str.strip, f.readline().split(',')))

        for line in self.parse_config_file(COLUMN_CONFIG_PATH, self.dataset_name):
            column_key, user_key = tuple(map(str.strip, line.strip().split("=")))
            self.__column_index[column_key] = split_column_line.index(user_key)

        for idx in self.__column_index.values():
            if idx is None:
                raise Exception("\"config/flow_column_map.txt\" is not correctly set")

    def __load_inside_ip_set(self):
        for cidr_subnet in self.parse_config_file(INSIDE_CONFIG_PATH, self.dataset_name):
            cidr_subnet = cidr_subnet.strip()
            self.__inside_ip_set |= subnet_to_ip_set(cidr_subnet)

    @classmethod
    def parse_config_file(cls, file_path: str, target: str) -> List[str]:
        with open(file_path, 'r') as f:
            lines = list(map(str.strip, f.readlines()))

        content_list = []

        is_find = False
        for line in lines:
            if not is_find:
                if line == '[' + target + ']':
                    is_find = True
            else:
                if line.startswith('[') and line.endswith(']'):
                    break
                if line:
                    content_list.append(line)
        return content_list
