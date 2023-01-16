import os
import torch
from torch.utils.data import DataLoader

from config import Config
from component.ip_profiling import Profiler
from component.feature import FeatureExtractor
from component import rfre
from ml.dataset import RFREDataset
from ml.train import trainer, tester
from experiments.stats import save_stats
from experiments.scatter import draw_rce_scatter
from utils import TQDM


def train(config: Config):
    print("############################")
    print("#    start train phase     #")
    print("############################")
    profiler = Profiler(config)
    fe = FeatureExtractor()

    for train_dir in config.train_dir_list:
        profiler.profile(train_dir, benign_only=True)
        for profile in TQDM(profiler, desc='extract features from profiles'):
            fe.extract(profile)

    rfre_feature_matrix = rfre.encode(fe.feature_matrix, fe.feature_list, config)
    dataset = RFREDataset(rfre_feature_matrix, fe.profile_key_list)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    trainer(config, dataloader)


def test(config: Config):
    print("############################")
    print("# start test phase  #######")
    print("############################")
    with open(config.path.model_path, 'rb') as f:
        model = torch.load(f).eval()

    for test_dir in config.test_dir_list:
        print(f"\nCurrent testing -> \"{test_dir}\"")
        profiler = Profiler(config)
        profiler.profile(test_dir)

        fe = FeatureExtractor()
        for profile in TQDM(profiler, desc='extract features from profiles'):
            fe.extract(profile)

        rfre_feature_matrix = rfre.encode(fe.feature_matrix, fe.feature_list, config)
        dataset = RFREDataset(rfre_feature_matrix, fe.profile_key_list)
        dataloader = DataLoader(dataset, batch_size=config.batch_size)

        result_csv = tester(model, dataloader)
        result_csv_path = os.path.join(config.path.result_dir, os.path.split(test_dir)[1] + ".csv")
        with open(result_csv_path, 'w') as f:
            for line in result_csv:
                f.write(line + "\n")


if __name__ == '__main__':
    _config = Config('CICIDS2017')
    train(_config)
    test(_config)
    save_stats(_config)
    draw_rce_scatter(_config)
