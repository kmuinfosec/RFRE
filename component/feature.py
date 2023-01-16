import numpy as np
from ipaddress import IPv4Address

from component.profile import Profile


class FeatureExtractor:
    feature_list = ['target_ip', 'target_key_port', 'opposite_key_port', 'card_target_port', 'card_opposite_ip',
                    'card_opposite_port',
                    'sum_target_pkts', 'sum_opposite_pkts', 'sum_target_bytes', 'sum_opposite_bytes', 'sum_dur',
                    'avg_target_pkts', 'avg_opposite_pkts', 'avg_target_bytes', 'avg_opposite_bytes', 'avg_dur',
                    'std_target_pkts', 'std_opposite_pkts', 'std_target_bytes', 'std_opposite_bytes', 'std_dur']

    feature_func_map = {
        'target_ip':
            lambda x: str(IPv4Address(int(IPv4Address(x.target_ip)) >> (32 - 32) << (32 - 32))),
        'target_key_port':
            lambda x: sorted(
                zip(*np.unique(x['target_port'], return_counts=True)), key=lambda y: y[1], reverse=True)[0][0],
        'opposite_key_port':
            lambda x: sorted(
                zip(*np.unique(x['opposite_port'], return_counts=True)), key=lambda y: y[1], reverse=True)[0][0],
        'card_target_port':
            lambda x: len(set(x['target_port'])),
        'card_opposite_ip':
            lambda x: len(set(x['opposite_ip'])),
        'card_opposite_port':
            lambda x: len(set(x['opposite_port'])),
        'sum_target_pkts':
            lambda x: np.sum(x['target_pkts']),
        'sum_opposite_pkts':
            lambda x: np.sum(x['opposite_pkts']),
        'sum_target_bytes':
            lambda x: np.sum(x['target_bytes']),
        'sum_opposite_bytes':
            lambda x: np.sum(x['opposite_bytes']),
        'sum_dur':
            lambda x: np.sum(x['duration']),
        'avg_target_pkts':
            lambda x: np.mean(x['target_pkts']),
        'avg_opposite_pkts':
            lambda x: np.mean(x['opposite_pkts']),
        'avg_target_bytes':
            lambda x: np.mean(x['target_bytes']),
        'avg_opposite_bytes':
            lambda x: np.mean(x['opposite_bytes']),
        'avg_dur':
            lambda x: np.mean(x['duration']),
        'std_target_pkts':
            lambda x: np.std(x['target_pkts']),
        'std_opposite_pkts':
            lambda x: np.std(x['opposite_pkts']),
        'std_target_bytes':
            lambda x: np.std(x['target_bytes']),
        'std_opposite_bytes':
            lambda x: np.std(x['opposite_bytes']),
        'std_dur':
            lambda x: np.std(x['duration'])
    }

    def __init__(self):
        self.__feature_matrix = [[] for _ in range(len(self.feature_list))]
        self.__profile_key_list = []

    def extract(self, profile: Profile):
        for i, feature in enumerate(self.feature_list):
            self.__feature_matrix[i].append(FeatureExtractor.feature_func_map[feature](profile))
        self.__profile_key_list.append(profile.profile_key)

    def debug(self) -> str:
        ret_str = self.__str_feature_list(profile_key=True) + '\n'
        for i, profile_key in enumerate(self.__profile_key_list):
            ret_str += profile_key+','+self.__str_feature_vector(i) + '\n'
        return ret_str

    def get_info(self, num_vectors=5) -> str:
        ret_str = '####################################################################################\n'
        ret_str += '# Feature Info (total vectors: {})\n'.format(len(self.profile_key_list))
        ret_str += '# \n'
        ret_str += '# Feature list\n'
        ret_str += '# {}\n'.format(self.__str_feature_list())
        ret_str += '# \n'
        ret_str += '# First {} vectors\n'.format(num_vectors)
        for i in range(num_vectors):
            ret_str += '# {}\n'.format(self.__str_feature_vector(i))
        ret_str += '####################################################################################\n'
        return ret_str

    def __str__(self) -> str:
        return self.get_info()

    def __str_feature_list(self, profile_key=False) -> str:
        ret_str = ''
        if profile_key:
            ret_str += 'profile_key,'
        ret_str += ','.join(self.feature_list)
        return ret_str

    def __str_feature_vector(self, index: int) -> str:
        ret_str = ''
        feature_vector = []
        for j, feature in enumerate(self.feature_list):
            feature_vector.append(str(self.feature_matrix[j][index]))
        ret_str += ','.join(feature_vector)
        return ret_str

    @property
    def feature_matrix(self):
        return self.__feature_matrix

    @property
    def profile_key_list(self):
        return self.__profile_key_list
