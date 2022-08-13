import sys
import os
from tqdm import tqdm

from utils import load_inside_ip_set, get_column_idx_table, get_time_window


def grouping_by_outer_ip(
        netflow_list: list, column_idx_table: dict, inside_ip_set: set, time_window: int, benign_only=False):

    group = {}
    for netflow in netflow_list:
        netflow = netflow.strip()
        split_netflow = netflow.split(",")
        source_ip = split_netflow[column_idx_table['source_ip']]
        destination_ip = split_netflow[column_idx_table['destination_ip']]
        start_time = split_netflow[column_idx_table['flow_start_time']]
        if benign_only and split_netflow[column_idx_table['label']].lower() != 'benign':
            continue

        window_start, window_end = get_time_window(start_time, time_window)
        key = "{}_"+window_start+'_'+window_end
        if source_ip in inside_ip_set:
            key = key.format(destination_ip)
        elif destination_ip in inside_ip_set:
            key = key.format(source_ip)
        else:
            continue
        if key not in group:
            group[key] = []
        group[key].append(netflow)

    return group


def profile(config, netflow_dir_path, train_mode=False):
    inside_ip_set = load_inside_ip_set(config['inside_list_path'])
    column_idx_table = None
    netflow_group_table = {}
    for netflow_file_name in tqdm(os.listdir(netflow_dir_path), desc='ip profiling', ascii=True, file=sys.stdout):
        netflow_file_path = os.path.join(netflow_dir_path, netflow_file_name)
        with open(netflow_file_path, 'r') as f:
            column_line = f.readline().strip()
            if column_idx_table is None:
                column_idx_table = get_column_idx_table(column_line, config['netflow_column_map_path'])
            netflow_list = f.readlines()

        for key, netflow_list in grouping_by_outer_ip(
                netflow_list, column_idx_table, inside_ip_set, config['time_window'], benign_only=train_mode).items():
            if key not in netflow_group_table:
                netflow_group_table[key] = []
            netflow_group_table[key] += netflow_list
    return netflow_group_table, column_idx_table

