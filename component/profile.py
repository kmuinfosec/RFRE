class Profile:

    attr_list = ['target_ip', 'target_port', 'opposite_ip', 'opposite_port', 'duration',
                 'target_pkts', 'opposite_pkts', 'target_bytes', 'opposite_bytes']

    attr_typing_map = {
        'target_ip': lambda x: str(x),
        'target_port': lambda x: int(float(x)),
        'opposite_ip': lambda x: str(x),
        'opposite_port': lambda x: int(float(x)),
        'duration': lambda x: float(x),
        'target_pkts': lambda x: int(float(x)),
        'opposite_pkts': lambda x: int(float(x)),
        'target_bytes': lambda x: int(float(x)),
        'opposite_bytes': lambda x: int(float(x))
    }

    def __init__(self, profile_key):
        self.__profile_key = profile_key
        self.__flow_cnt = 0
        self.__table = {}
        for attr in self.attr_list:
            self.__table[attr] = []

    def add(self, attr_dict: dict):
        for attr in self.attr_list:
            value = self.attr_typing_map[attr](attr_dict[attr])
            self.__table[attr].append(value)
        self.__flow_cnt += 1

    def debug(self):
        ret_str = self.__str_attr() + '\n'
        for idx in range(self.__flow_cnt):
            ret_str += self.__str_row(idx) + '\n'
        return ret_str

    def __str_attr(self) -> str:
        return ",".join(self.attr_list)

    def __str_row(self, idx: int) -> str:
        print_row_values = []
        for attr in self.attr_list:
            print_row_values.append(str(self.__table[attr][idx]))
        return ",".join(print_row_values)

    def get_info(self, num_flow=5) -> str:
        ret_str = '####################################################################################\n'
        ret_str += '# Profile Info (key: {}, total flows: {})\n'.format(self.profile_key, self.__flow_cnt)
        ret_str += '# \n'
        ret_str += '# Attribute list\n'
        ret_str += '# {}\n'.format(self.__str_attr())
        ret_str += '# \n'
        ret_str += '# First {} flows\n'.format(min(self.__flow_cnt, num_flow))
        for i in range(min(self.__flow_cnt, num_flow)):
            ret_str += '# {}\n'.format(self.__str_row(i))
        ret_str += '####################################################################################\n'
        return ret_str

    def __str__(self) -> str:
        return self.get_info()

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, attr) -> list:
        return self.table[attr]

    @property
    def table(self) -> dict:
        return self.__table

    @property
    def profile_key(self) -> str:
        return self.__profile_key

    @property
    def target_ip(self) -> str:
        return self.profile_key.split('_')[0]
