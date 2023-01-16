import sys
import os
from tqdm import tqdm

from component.profile import Profile
from config import Config
from utils import get_time_window


class Profiler:

    attr_map = {'target_ip': 'dip', 'target_port': 'dport', 'opposite_ip': 'sip', 'opposite_port': 'sport',
                'duration': 'duration', 'target_pkts': 'out_packets', 'opposite_pkts': 'in_packets',
                'target_bytes': 'out_bytes', 'opposite_bytes': 'in_bytes'}

    attr_map_inv = {'target_ip': 'sip', 'target_port': 'sport', 'opposite_ip': 'dip', 'opposite_port': 'dport',
                    'duration': 'duration', 'target_pkts': 'in_packets', 'opposite_pkts': 'out_packets',
                    'target_bytes': 'in_bytes', 'opposite_bytes': 'out_bytes'}

    def __init__(self, config: Config):
        self.__profile_cnt = 0
        self.__profile_index = {}
        self.__profile_index_inv = {}
        self.__profile_list = []
        self.__inside_ip_set = config.inside_ip_set
        self.__time_window = config.time_window
        self.__profiling_target = config.profiling_target
        self.__column_index = config.column_index

    def profile(self, flow_dir: str, benign_only: bool = False):
        for flow_file_name in tqdm(os.listdir(flow_dir), desc='ip profiling', ascii=True, file=sys.stdout):
            flow_file_path = os.path.join(flow_dir, flow_file_name)
            with open(flow_file_path, 'r') as f:
                f.readline()    # pass column row
                flow_list = f.readlines()

            for flow in flow_list:
                flow = list(map(str.strip, flow.strip().split(",")))
                if benign_only and flow[self.__column_index['label']].lower() != 'benign':
                    continue
                self.add_flow(flow)

    def add_flow(self, flow: list):
        sip, dip = flow[self.__column_index['sip']], flow[self.__column_index['dip']]
        start_time = flow[self.__column_index['time_start']]
        window_start, window_end = get_time_window(start_time, self.__time_window)

        attr_map = {}
        profile_key = '{}_{}_{}'
        if self.profiling_target == 'dip':
            if sip in self.__inside_ip_set:
                attr_map = self.attr_map
                profile_key = profile_key.format(dip, window_start, window_end)
            elif dip in self.__inside_ip_set:
                attr_map = self.attr_map_inv
                profile_key = profile_key.format(sip, window_start, window_end)
            else:
                return
        elif self.profiling_target == 'sip':
            if dip in self.__inside_ip_set:
                attr_map = self.attr_map_inv
                profile_key = profile_key.format(sip, window_start, window_end)
            elif sip in self.__inside_ip_set:
                attr_map = self.attr_map
                profile_key = profile_key.format(dip, window_start, window_end)
            else:
                return

        attr_dict = {}
        for attr, column in attr_map.items():
            attr_dict[attr] = flow[self.__column_index[column]]
        self.__add_profile(profile_key)
        self[profile_key].add(attr_dict)

    def get_profile_key(self, index: int) -> str:
        if index not in self.__profile_index_inv:
            raise IndexError(f"There is no '{index}' chunk")
        return self.__profile_index_inv[index]

    def get_profile(self, profile_key: str) -> Profile:
        return self[profile_key]

    def get_profile_by_index(self, index: int) -> Profile:
        return self[self.get_profile_key(index)]

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
    def profiling_target(self) -> str:
        return self.__profiling_target

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
