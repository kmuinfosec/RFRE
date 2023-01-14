import sys
import os
from tqdm import tqdm

from component.profile import Profile
from config import Config
from utils import get_time_window


class Profiler:

    def __init__(self, config: Config):
        self.__profile_cnt = 0
        self.__profile_index = {}
        self.__profile_index_inv = {}
        self.__profile_list = []
        self.__inside_ip_set = config.inside_ip_set
        self.__time_window = config.time_window
        self.__flow_column_map = config.flow_column_map

    def profile(self, flow_dir: str, benign_only: bool = False):
        for flow_file_name in tqdm(os.listdir(flow_dir), desc='ip profiling', ascii=True, file=sys.stdout):
            flow_file_path = os.path.join(flow_dir, flow_file_name)
            with open(flow_file_path, 'r') as f:
                column_line = f.readline().strip()
                flow_list = f.readlines()

            column_idx_map = get_column_idx_map(column_line, self.__flow_column_map)

            for flow in flow_list:
                flow = flow.strip().split(",")
                sip, dip = flow[column_idx_map['sip']], flow[column_idx_map['dip']]
                start_time = flow[column_idx_map['time_start']]
                window_start, window_end = get_time_window(start_time, self.__time_window)

                if benign_only and flow[column_idx_map['label']].lower() != 'benign':
                    continue

                if sip in self.__inside_ip_set:
                    profile_key = f'{flow[column_idx_map["dip"]]}_{window_start}_{window_end}'
                    self.add_flow(profile_key, flow, column_idx_map)
                elif dip in self.__inside_ip_set:
                    profile_key = f'{flow[column_idx_map["sip"]]}_{window_start}_{window_end}'
                    self.add_flow(profile_key, flow, column_idx_map, by_src=False)
                else:
                    continue

    def add_flow(self, profile_key: str, flow: list, column_idx_map: dict, by_src=True):
        self.__add_profile(profile_key)
        self[profile_key].add(flow, column_idx_map, by_src)

    def get_profile_key(self, index: int) -> str:
        if index not in self.__profile_index_inv:
            raise IndexError(f"There is no '{index}' chunk")
        return self.__profile_index_inv[index]

    def __add_profile(self, profile_key):
        if profile_key not in self:
            new_pf = Profile(profile_key)
            self.__profile_list.append(new_pf)
            self.__profile_index[profile_key] = self.__profile_cnt
            self.__profile_index_inv[self.__profile_cnt] = profile_key
            self.__profile_cnt += 1

    def __len__(self) -> int:
        return self.__profile_cnt

    def __contains__(self, profile_key: str) -> bool:
        return profile_key in self.__profile_index

    def __getitem__(self, profile_key: str) -> Profile:
        if profile_key not in self.__profile_index:
            raise IndexError(f"{profile_key} is not in container")
        return self.profile_list[self.__profile_index[profile_key]]

    def __iter__(self):
        return ProfilerIterator(self)

    @property
    def profile_list(self):
        return self.__profile_list


class ProfilerIterator:

    def __init__(self, profiler: Profiler):
        self.__profiler = profiler
        self.index = 0

    def __next__(self):
        if self.index >= len(self.__profiler):
            raise StopIteration()
        ret_profile = self.__profiler[self.__profiler.get_profile_key(self.index)]
        self.index += 1

        return ret_profile

    def __iter__(self):
        return self


def get_column_idx_map(column_line, netflow_column_map):
    column_idx_table = {}
    split_column_line = column_line.split(",")
    for column_key, user_key in netflow_column_map.items():
        column_idx = split_column_line.index(user_key)
        column_idx_table[column_key] = column_idx
    return column_idx_table
