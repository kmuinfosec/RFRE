import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.mixture import BayesianGaussianMixture
from typing import List

from config import Config


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

    def transform(self, data) -> np.ndarray:
        new_data = np.zeros(len(data), dtype=np.float_)
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

    def fit_transform(self, data) -> np.ndarray:
        self.fit(data)
        return self.transform(data)


def vgm_clustering(feature_matrix: List[list], feature_list: List[str], vgm_path: str) -> List[np.ndarray]:
    vgm_table = {}
    if os.path.isfile(vgm_path):
        with open(vgm_path, 'rb') as f:
            vgm_table = pickle.load(f)

    vgm_feature_matrix = []
    for i, feature in tqdm(enumerate(feature_list), desc="vgm clustering", ascii=True, file=sys.stdout, total=len(feature_list)):
        if 'ip' in feature or 'port' in feature:
            vgm_feature_matrix.append(feature_matrix[i])
            continue
        feature_column = np.array(feature_matrix[i]).reshape(-1, 1)
        if feature not in vgm_table:
            vgm = BayesianGaussianMixture(n_components=10, max_iter=2000)
            vgm.fit(feature_column)
            vgm_table[feature] = vgm
        vgm = vgm_table[feature]
        pred = vgm.predict(feature_column)
        vgm_feature_matrix.append(pred)

    if not os.path.isfile(vgm_path):
        with open(vgm_path, 'wb') as f:
            pickle.dump(vgm_table, f)

    return vgm_feature_matrix


def rfre_encoding(vgm_feature_matrix: List[np.ndarray], feature_list: List[str], rfre_encoder_path: str) -> List[np.ndarray]:
    rfre_encoder_table = {}
    if os.path.isfile(rfre_encoder_path):
        with open(rfre_encoder_path, 'rb') as f:
            rfre_encoder_table = pickle.load(f)

    rfre_feature_matrix = []
    for i, feature in tqdm(enumerate(feature_list), desc="rfre encoding", ascii=True, file=sys.stdout, total=len(feature_list)):
        if feature not in rfre_encoder_table:
            rfre_encoder = RelativeFrequencyRankEncoder()
            rfre_encoder.fit(vgm_feature_matrix[i])
            rfre_encoder_table[feature] = rfre_encoder
        rfre_encoder = rfre_encoder_table[feature]
        rfre_feature_matrix.append(rfre_encoder.transform(vgm_feature_matrix[i]))

    if not os.path.isfile(rfre_encoder_path):
        with open(rfre_encoder_path, 'wb') as f:
            pickle.dump(rfre_encoder_table, f)

    return rfre_feature_matrix


def encode(feature_matrix: List[list], feature_list: List[str], config: Config) -> List[np.ndarray]:
    vgm_feature_matrix = vgm_clustering(feature_matrix, feature_list, config.path.vgm_path)
    rfre_feature_matrix = rfre_encoding(vgm_feature_matrix, feature_list, config.path.rfre_encoder_path)
    return rfre_feature_matrix
