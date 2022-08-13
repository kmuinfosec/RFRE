import os
import random
import numpy as np
import torch

from ipaddress import IPv4Address
from datetime import datetime
from typing import Tuple


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)



def get_column_idx_table(column_line, netflow_column_map_path):
    column_idx_table = {}
    user_column_map = get_user_column_map(netflow_column_map_path)
    split_column_line = column_line.split(",")
    for column_key, user_key in user_column_map.items():
        column_idx = split_column_line.index(user_key)
        column_idx_table[column_key] = column_idx
    return column_idx_table


def load_inside_ip_set(inside_list_path):
    inside_ip_set = set()
    with open(inside_list_path, 'r') as f:
        for cidr_subnet in f.readlines():
            cidr_subnet = cidr_subnet.strip()
            inside_ip_set |= subnet_to_ip_set(cidr_subnet)
    return inside_ip_set


def subnet_to_ip_set(cidr_subnet):
    ip_set = set()
    if '/' in cidr_subnet:
        subnet, cidr = cidr_subnet.split('/')
        start_address = int(IPv4Address(subnet))
        end_address = int(IPv4Address(subnet)) + 2**(32-int(cidr))-1
        for subnet_address in range(start_address, end_address+1):
            ip_set.add(str(IPv4Address(subnet_address)))
    else:
        if '.' in cidr_subnet and len(cidr_subnet.split('.')) == 4:
            ip_set.add(cidr_subnet)
    return ip_set


def get_user_column_map(netflow_column_map_config_path):
    user_column_map = {
        'source_ip': None,
        'source_port': None,
        'destination_ip': None,
        'destination_port': None,
        'flow_start_time': None,
        'flow_end_time': None,
        'flow_duration': None,
        'number_of_in_packets': None,
        'number_of_out_packets': None,
        'size_of_in_bytes': None,
        'size_of_out_bytes': None,
        'label': None
    }
    with open(netflow_column_map_config_path, 'r') as f:
        for line in f.readlines():
            column_key, user_key = line.strip().split("=")
            user_column_map[column_key] = user_key

    for column_key, user_key in user_column_map.items():
        if user_key is None:
            raise Exception("\"config/netflow_column_map.txt\" is not correctly set")

    return user_column_map


def get_time_window(start_time: str, time_window: int) -> Tuple[str, str]:
    datetime_format = '%Y-%m-%d %H:%M:%S'
    ts_timestamp = datetime.strptime(start_time, datetime_format).timestamp()
    window_start = ts_timestamp - (ts_timestamp % time_window)
    window_end = window_start + time_window
    window_start_str = datetime.fromtimestamp(window_start).strftime(datetime_format)
    window_end_str = datetime.fromtimestamp(window_end).strftime(datetime_format)

    return window_start_str, window_end_str


def get_device(cuda_num=0):
    return f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu'


def format_date(precision: int) -> str:
    """
    :param precision: year:0, month:1, day:2, hour:3, minute:4, second:5
    :return: formatted datetime string
    """
    format_order = ["%Y", "%m", "%d", "%H", "%M", "%S"]
    format_str = ""
    for i, fmt in enumerate(format_order):
        if i == precision + 1:
            break
        else:
            format_str += fmt
            if i == 2:
                format_str += '_'

    return datetime.today().strftime(format_str)


def get_config(outcome_name):
    base_dir_path = os.path.dirname(os.path.abspath(__file__))
    config_dir_path = os.path.join(base_dir_path, 'config')
    inside_list_path = os.path.join(config_dir_path, 'inside_list.txt')
    netflow_column_map_path = os.path.join(config_dir_path, 'netflow_column_map.txt')
    train_dir_path = os.path.join(config_dir_path, 'train_directory.txt')
    test_dir_list_path = os.path.join(config_dir_path, 'test_directory_list.txt')

    with open(train_dir_path, 'r') as f:
        train_netflow_dir = list(map(str.strip, f.readlines()))[0]

    with open(test_dir_list_path, 'r') as f:
        test_netflow_dir_list = list(map(str.strip, f.readlines()))

    outcome_dir_path = os.path.join(base_dir_path, 'result')
    if not os.path.isdir(outcome_dir_path):
        os.mkdir(outcome_dir_path)
    current_outcome_dir_path = os.path.join(outcome_dir_path, outcome_name)
    if not os.path.isdir(current_outcome_dir_path):
        os.mkdir(current_outcome_dir_path)
    config = {
        'inside_list_path': inside_list_path,
        'netflow_column_map_path': netflow_column_map_path,
        'outcome_dir_path': current_outcome_dir_path,
        'train_netflow_dir': train_netflow_dir,
        'test_netflow_dir_list': test_netflow_dir_list,
        'time_window': 300,
        'interface': 32,
        'epochs': 200,
        'batch_size': 256
    }
    return config
