import os
import sys

from torch.cuda import is_available
from tqdm import tqdm
from datetime import datetime
from ipaddress import IPv4Address
from typing import Set, Tuple


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


def subnet_to_ip_set(cidr_subnet: str) -> Set[str]:
    ip_set = set()
    if '/' in cidr_subnet:
        subnet, cidr = cidr_subnet.split('/')
        start_address = int(IPv4Address(subnet))
        end_address = int(IPv4Address(subnet)) + 2 ** (32 - int(cidr)) - 1
        for subnet_address in range(start_address, end_address + 1):
            ip_set.add(str(IPv4Address(subnet_address)))
    else:
        if '.' in cidr_subnet and len(cidr_subnet.split('.')) == 4:
            ip_set.add(cidr_subnet)
    return ip_set


def get_device(cuda_num=0):
    return f'cuda:{cuda_num}' if is_available() else 'cpu'


def get_time_window(start_time: str, time_window: int) -> Tuple[str, str]:
    datetime_format = '%Y-%m-%d %H:%M:%S'
    ts_timestamp = datetime.strptime(start_time, datetime_format).timestamp()
    window_start = ts_timestamp - (ts_timestamp % time_window)
    window_end = window_start + time_window
    window_start_str = datetime.fromtimestamp(window_start).strftime(datetime_format)
    window_end_str = datetime.fromtimestamp(window_end).strftime(datetime_format)

    return window_start_str, window_end_str


def create_directory(directory_path: str, recursive=False):
    if not os.path.isdir(directory_path):
        if recursive:
            os.makedirs(directory_path)
        else:
            os.mkdir(directory_path)


def TQDM(iterable, desc=''):
    for elem in tqdm(iterable, ascii=True, desc=desc, file=sys.stdout):
        yield elem
