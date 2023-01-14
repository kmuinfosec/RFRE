import torch
import numpy as np
from torch.utils.data.dataloader import Dataset
from typing import List


class RFREDataset(Dataset):

    def __init__(self, rfre_feature_matrix: List[np.ndarray], target_ip_list: List[str]):
        super(RFREDataset, self).__init__()
        self.rfre_feature_matrix = rfre_feature_matrix
        self.target_ip_list = target_ip_list

    def __len__(self):
        return len(self.target_ip_list)

    def __getitem__(self, idx):
        feature_vector = []
        for feature_column in self.rfre_feature_matrix:
            feature_vector.append(feature_column[idx])

        x = torch.FloatTensor(feature_vector)

        return x, self.target_ip_list[idx]
