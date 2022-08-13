import sys
import numpy as np
from ipaddress import IPv4Address
from tqdm import tqdm


def extract_original_features(group_key: str, netflow_list: list, column_idx_table: dict, interface: int):
    key_ip = group_key.split("_")[0]
    inner_ip_list, inner_port_list, outer_port_list, td_list = [], [], [], []
    inner_bytes_list, outer_bytes_list, inner_pkts_list, outer_pkts_list = [], [], [], []
    for netflow in netflow_list:
        split_netflow = netflow.split(",")
        if key_ip == split_netflow[column_idx_table['source_ip']]:
            inner_ip_list.append(split_netflow[column_idx_table['destination_ip']])
            inner_port_list.append(int(float(split_netflow[column_idx_table['destination_port']])))
            outer_port_list.append(int(float(split_netflow[column_idx_table['source_port']])))
            td_list.append(float(split_netflow[column_idx_table['flow_duration']]))
            inner_pkts_list.append(int(float(split_netflow[column_idx_table['number_of_out_packets']])))
            outer_pkts_list.append(int(float(split_netflow[column_idx_table['number_of_in_packets']])))
            inner_bytes_list.append(int(float(split_netflow[column_idx_table['size_of_out_bytes']])))
            outer_bytes_list.append(int(float(split_netflow[column_idx_table['size_of_in_bytes']])))
        else:
            inner_ip_list.append(split_netflow[column_idx_table['source_ip']])
            inner_port_list.append(int(float(split_netflow[column_idx_table['source_port']])))
            outer_port_list.append(int(float(split_netflow[column_idx_table['destination_port']])))
            td_list.append(float(split_netflow[column_idx_table['flow_duration']]))
            inner_pkts_list.append(int(float(split_netflow[column_idx_table['number_of_in_packets']])))
            outer_pkts_list.append(int(float(split_netflow[column_idx_table['number_of_out_packets']])))
            inner_bytes_list.append(int(float(split_netflow[column_idx_table['size_of_in_bytes']])))
            outer_bytes_list.append(int(float(split_netflow[column_idx_table['size_of_out_bytes']])))

    feature_table = {'key_ip': str(
        IPv4Address(int(IPv4Address(key_ip)) >> (32 - interface) << (32 - interface))),
        'key_inner_port': sorted(
            zip(*np.unique(inner_port_list, return_counts=True)), key=lambda x: x[1], reverse=True)[0][0],
        'key_outer_port': sorted(
            zip(*np.unique(outer_port_list, return_counts=True)), key=lambda x: x[1], reverse=True)[0][0],
        'card_inner_ip': len(set(inner_ip_list)),
        'card_inner_port': len(set(inner_port_list)),
        'card_outer_port': len(set(outer_port_list)),
        'sum_inner_pkts': np.sum(inner_pkts_list),
        'sum_outer_pkts': np.sum(outer_pkts_list),
        'sum_inner_bytes': np.sum(inner_bytes_list),
        'sum_outer_bytes': np.sum(outer_bytes_list),
        'sum_dur': np.sum(td_list),
        'avg_inner_pkts': np.mean(inner_pkts_list),
        'avg_outer_pkts': np.mean(outer_pkts_list),
        'avg_inner_bytes': np.mean(inner_bytes_list),
        'avg_outer_bytes': np.mean(outer_bytes_list),
        'avg_dur': np.mean(td_list),
        'std_inner_pkts': np.std(inner_pkts_list),
        'std_outer_pkts': np.std(outer_pkts_list),
        'std_inner_bytes': np.std(inner_bytes_list),
        'std_outer_bytes': np.std(outer_bytes_list),
        'std_dur': np.std(td_list)
    }

    return feature_table


def extract(netflow_group_table: dict, column_idx_table: dict, interface: int):
    original_feature_table = {}
    group_key_list = []
    for group_key, netflow_list in tqdm(netflow_group_table.items(), desc='extract original feature vectors', ascii=True, file=sys.stdout):
        for k, v in extract_original_features(group_key, netflow_list, column_idx_table, interface).items():
            if k not in original_feature_table:
                original_feature_table[k] = []
            original_feature_table[k].append(v)
        group_key_list.append(group_key)
    return original_feature_table, group_key_list
