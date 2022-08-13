import torch
from torch.utils.data.dataloader import Dataset


class CustomDataset(Dataset):

    def __init__(self, rfre_feature_table: dict, group_key_list: list):
        super(CustomDataset, self).__init__()
        self.rfre_feature_table = rfre_feature_table
        self.group_key_list = group_key_list

    def __len__(self):
        return len(self.group_key_list)

    def __getitem__(self, idx):
        feature_vector = []
        for k in self.rfre_feature_table.keys():
            feature_vector.append(self.rfre_feature_table[k][idx])

        x = torch.FloatTensor(feature_vector)

        return x, self.group_key_list[idx]
