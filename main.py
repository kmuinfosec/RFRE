import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from component.ip_profiling import profile
from component.features import extract
from component.rfre import encode
from ml.dataset import CustomDataset
from ml.train import trainer
from experiments.stats import save_stats
from utils import get_config, format_date


def train(config):
    netflow_group_table, column_idx_table = profile(config, config['train_netflow_dir'], train_mode=True)
    original_feature_table, group_key_list = extract(netflow_group_table, column_idx_table, config['interface'])
    rfre_feature_table = encode(original_feature_table, config['outcome_dir_path'])

    dataset = CustomDataset(rfre_feature_table, group_key_list)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    trainer(dataloader, config['epochs'], config['outcome_dir_path'])


def test(config):
    with open(os.path.join(config['outcome_dir_path'], 'model.pth'), 'rb') as f:
        model = torch.load(f).eval()

    for test_netflow_dir in config['test_netflow_dir_list']:
        print("\nstart testing \"{}\" directory".format(test_netflow_dir))
        netflow_group_table, column_idx_table = profile(config, test_netflow_dir)
        original_feature_table, group_key_list = extract(netflow_group_table, column_idx_table, config['interface'])
        rfre_feature_table = encode(original_feature_table, config['outcome_dir_path'])

        dataset = CustomDataset(rfre_feature_table, group_key_list)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'])

        result_csv = ['key,rce']
        with torch.inference_mode():
            for batch, group_key_batch in tqdm(dataloader, ascii=True, file=sys.stdout, desc="testing"):
                recons = model(batch)
                rces = model.mse(batch, recons, reduction='none')
                for i in range(len(rces)):
                    rce = rces[i].item()
                    key = group_key_batch[i]
                    result_csv.append(",".join([key, str(rce)]))

        result_dir_path = os.path.join(config['outcome_dir_path'], 'result')
        if not os.path.isdir(result_dir_path):
            os.mkdir(result_dir_path)
        result_csv_path = os.path.join(result_dir_path, os.path.split(test_netflow_dir)[1]+".csv")
        with open(result_csv_path, 'w') as f:
            for line in result_csv:
                f.write(line+"\n")


if __name__ == '__main__':
    config = get_config(format_date(6))
    train(config)
    test(config)
    save_stats(config['test_netflow_dir_list'], config)
