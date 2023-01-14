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

    attr_map = {'target_ip': 'dip', 'target_port': 'dport', 'opposite_ip': 'sip', 'opposite_port': 'sport',
                'duration': 'duration', 'target_pkts': 'out_packets', 'opposite_pkts': 'in_packets',
                'target_bytes': 'out_bytes', 'opposite_bytes': 'in_bytes'}

    attr_map_inv = {'target_ip': 'sip', 'target_port': 'sport', 'opposite_ip': 'dip', 'opposite_port': 'dport',
                    'duration': 'duration', 'target_pkts': 'in_packets', 'opposite_pkts': 'out_packets',
                    'target_bytes': 'in_bytes', 'opposite_bytes': 'out_bytes'}

    def __init__(self, profile_key):
        self.__profile_key = profile_key
        self.__flow_cnt = 0
        self.__table = {}
        for attr in self.attr_list:
            self.__table[attr] = []

    def add(self, flow, column_idx_map, by_src=True):
        target_map = self.attr_map if by_src else self.attr_map_inv
        for attr, column in target_map.items():
            value = self.attr_typing_map[attr](flow[column_idx_map[column]])
            self.__table[attr].append(value)
        self.__flow_cnt += 1

    def print_attr(self):
        print(", ".join(self.attr_list))

    def print_row(self, idx: int):
        print_row_values = []
        for attr in self.attr_list:
            print_row_values.append(str(self.__table[attr][idx]))
        print(", ".join(print_row_values))

    def print_all(self):
        self.print_attr()
        for i in range(self.__flow_cnt):
            self.print_row(i)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, attr):
        return self.table[attr]

    @property
    def table(self):
        return self.__table

    @property
    def profile_key(self):
        return self.__profile_key

    @property
    def target_ip(self):
        return self.table['target_ip'][0]
