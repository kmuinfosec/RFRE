import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.mixture import BayesianGaussianMixture


class RelativeFrequencyRankEncoder:

    def __init__(self):
        self.encoding_table = None
        self.min_value = None

    def fit(self, data):
        # build frequency table
        freq_table = {}
        x_list, freq_list = np.unique(data, return_counts=True)
        for i, freq in enumerate(freq_list):
            if freq not in freq_table:
                freq_table[freq] = []
            freq_table[freq].append(x_list[i])

        # make encoding table
        self.encoding_table = {}

        no_of_ranking = len(list(freq_table.keys()))
        for ranking, freq in enumerate(sorted(freq_table)):
            for data in freq_table[freq]:
                self.encoding_table[data] = (ranking+1) / no_of_ranking

        self.min_value = min(self.encoding_table.values())

    def transform(self, data):
        new_data = np.zeros(len(data), dtype=np.float)
        if type(data) is list or len(data.shape) == 1:
            for idx in range(len(data)):
                if data[idx] in self.encoding_table:
                    new_data[idx] = self.encoding_table[data[idx]]
                else:
                    new_data[idx] = 0
        else:
            for idx in range(len(data)):
                if data[idx][0] in self.encoding_table:
                    new_data[idx] = self.encoding_table[data[idx][0]]
                else:
                    new_data[idx] = 0
        return new_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def vgm_clustering(original_feature_table: dict, outcome_dir_path: str):
    clustering_table_path = os.path.join(outcome_dir_path, 'clustering_table.pkl')
    clustering_table = {}
    if os.path.isfile(clustering_table_path):
        with open(clustering_table_path, 'rb') as f:
            clustering_table = pickle.load(f)

    for k, v in tqdm(original_feature_table.items(), desc="vgm clustering", ascii=True, file=sys.stdout):
        if 'key' in k or 'card' in k:
            continue
        v = np.array(v).reshape(-1, 1)
        if k not in clustering_table:
            vgm = BayesianGaussianMixture(n_components=10, max_iter=2000)
            vgm.fit(v)
            clustering_table[k] = vgm
        vgm = clustering_table[k]
        pred = vgm.predict(v)
        original_feature_table[k] = pred

    if not os.path.isfile(clustering_table_path):
        with open(clustering_table_path, 'wb') as f:
            pickle.dump(clustering_table, f)

    return original_feature_table


def rfre_encoding(clustered_feature_table: dict, outcome_dir_path: str):
    rfre_feature_table = {}

    rfre_encoder_table_path = os.path.join(outcome_dir_path, 'rfre_encder_table.pkl')
    rfre_encoder_table = {}
    if os.path.isfile(rfre_encoder_table_path):
        with open(rfre_encoder_table_path, 'rb') as f:
            rfre_encoder_table = pickle.load(f)

    for k, v in tqdm(clustered_feature_table.items(), desc="rfre encoding", ascii=True, file=sys.stdout):
        if k not in rfre_encoder_table:
            rfre_encoder = RelativeFrequencyRankEncoder()
            rfre_encoder.fit(v)
            rfre_encoder_table[k] = rfre_encoder
        rfre_encoder = rfre_encoder_table[k]
        rfre_feature_table[k] = rfre_encoder.transform(v)

    if not os.path.isfile(rfre_encoder_table_path):
        with open(rfre_encoder_table_path, 'wb') as f:
            pickle.dump(rfre_encoder_table, f)

    return rfre_feature_table


def encode(original_feature_table: dict, outcome_dir_path: str):
    clustered_feature_table = vgm_clustering(original_feature_table, outcome_dir_path)
    rfre_feature_table = rfre_encoding(clustered_feature_table, outcome_dir_path)
    return rfre_feature_table
