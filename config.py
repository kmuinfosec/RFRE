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
INSIDE_CONFIG_PATH = os.path.join(CONFIG_DIR, 'inside_list.txt')
COLUMN_CONFIG_PATH = os.path.join(CONFIG_DIR, 'flow_column_map.txt')
TRAIN_CONFIG_PATH = os.path.join(CONFIG_DIR, 'train_dir.txt')
TEST_CONFIG_PATH = os.path.join(CONFIG_DIR, 'test_dir.txt')


class Config:

    def __init__(self, dataset_name: str, outcome_name: str = format_date(6)):
        self.__dataset_name = dataset_name
        self.__outcome_dir = os.path.join(OUTCOME_DIR, outcome_name)
        self.__result_dir = os.path.join(self.outcome_dir, 'result')
        self.__model_path = os.path.join(self.outcome_dir, 'model.pth')
        self.__model_txt_path = os.path.join(self.outcome_dir, 'model.txt')
        self.__threshold_path = os.path.join(self.outcome_dir, 'threshold.txt')
        self.__inside_ip_set = set()
        self.__train_dir_list = []
        self.__test_dir_list = []
        self.__time_window = 300
        self.__interface = 32
        self.__epochs = 250
        self.__batch_size = 256
        self.__flow_column_map = {
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

        self.build()
        self.show_info()

    def build(self):
        create_directory(self.outcome_dir, recursive=True)
        create_directory(self.result_dir, recursive=True)
        self.__load_inside_ip_set()
        self.__load_flow_column_map()
        self.__train_dir_list += self.parse_config_file(TRAIN_CONFIG_PATH, self.dataset_name)
        self.__test_dir_list += self.parse_config_file(TEST_CONFIG_PATH, self.dataset_name)

    def show_info(self):
        print('################################################################')
        print(f'# Dataset: {self.dataset_name}')
        print(f'# Outcome directory path: {self.outcome_dir}')
        print(f'# time window(s): {self.time_window}, interface: /{self.interface}')
        print(f'# epochs: {self.epochs}, batch size: {self.batch_size}')
        print('################################################################\n')

    @property
    def dataset_name(self) -> str:
        return self.__dataset_name

    @property
    def outcome_dir(self) -> str:
        return self.__outcome_dir

    @property
    def result_dir(self) -> str:
        return self.__result_dir

    @property
    def model_path(self) -> str:
        return self.__model_path

    @property
    def model_txt_path(self) -> str:
        return self.__model_txt_path

    @property
    def threshold_path(self) -> str:
        return self.__threshold_path

    @property
    def inside_ip_set(self) -> set:
        return self.__inside_ip_set

    @property
    def flow_column_map(self) -> dict:
        return self.__flow_column_map

    @property
    def train_dir_list(self) -> List[str]:
        return self.__train_dir_list

    @property
    def test_dir_list(self) -> List[str]:
        return self.__test_dir_list

    @property
    def time_window(self) -> int:
        return self.__time_window

    @property
    def interface(self) -> int:
        return self.__interface

    @property
    def epochs(self) -> int:
        return self.__epochs

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    def __load_flow_column_map(self):
        for line in self.parse_config_file(COLUMN_CONFIG_PATH, self.dataset_name):
            column_key, user_key = tuple(map(str.strip, line.strip().split("=")))
            self.__flow_column_map[column_key] = user_key

        for user_column in self.__flow_column_map.values():
            if user_column is None:
                raise Exception("\"config/flow_column_map.txt\" is not correctly set")

    def __load_inside_ip_set(self):
        for cidr_subnet in self.parse_config_file(INSIDE_CONFIG_PATH, self.dataset_name):
            cidr_subnet = cidr_subnet.strip()
            self.__inside_ip_set |= subnet_to_ip_set(cidr_subnet)

    @classmethod
    def parse_config_file(cls, file_path: str, dataset_name: str) -> List[str]:
        with open(file_path, 'r') as f:
            lines = list(map(str.strip, f.readlines()))

        content_list = []

        is_find = False
        for line in lines:
            if not is_find:
                if line == '[' + dataset_name + ']':
                    is_find = True
            else:
                if line.startswith('[') and line.endswith(']'):
                    break
                if line:
                    content_list.append(line)
        return content_list
