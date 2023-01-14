import os
import pandas as pd
from config import Config
from matplotlib import pyplot as plt
from utils import get_time_window, TQDM


FONT_SIZE = 30
POINT_SIZE = 100
NUM_COLORS = 20
cm = plt.get_cmap('tab20')


def load_result_df(config: Config):
    result_info = {}
    for file_name in os.listdir(config.result_dir):
        result_csv_path = os.path.join(config.result_dir, file_name)
        test_df = pd.read_csv(result_csv_path)
        for row in test_df.itertuples():
            key = row.key
            rce = row.rce
            result_info[key] = {'rce': rce, 'label': {}}

    for test_netflow_dir in TQDM(config.test_dir_list, desc="loading results for plotting"):
        for file_name in os.listdir(test_netflow_dir):
            file_path = os.path.join(test_netflow_dir, file_name)
            current_df = pd.read_csv(file_path)
            for row in current_df.itertuples():
                window_start, window_end = get_time_window(row.ts, config.time_window)
                if row.sa in config.inside_ip_set:
                    key_ip = row.da
                elif row.da in config.inside_ip_set:
                    key_ip = row.sa
                else:
                    continue
                key = f'{key_ip}_{window_start}_{window_end}'
                label = row.Label
                if label not in result_info[key]['label']:
                    result_info[key]['label'][label] = 0
                result_info[key]['label'][label] += 1

    for k, v in result_info.items():
        result_info[k]['label'] = sorted(v['label'].items(), key=lambda x: x[1], reverse=True)[0][0].lower()

    key_list = []
    rce_list = []
    label_list = []
    for k, v in result_info.items():
        key_list.append(k)
        rce_list.append(v['rce'])
        label_list.append(v['label'])

    result_df = pd.DataFrame()
    result_df['key'] = key_list
    result_df['rce'] = rce_list
    result_df['label'] = label_list

    return result_df


def draw_rce_scatter(config: Config):
    result_df = load_result_df(config)

    with open(config.threshold_path, 'r') as f:
        threshold = float(f.readlines()[-1].strip())

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    benign_df = result_df[result_df['label'] == 'benign']
    ax.scatter(benign_df.index, benign_df.rce, label='benign', alpha=0.5)
    for idx, label in enumerate(result_df.label.unique()):
        if label == 'benign':
            continue
        label_df = result_df[result_df['label'] == label]
        plt.scatter(label_df.index, label_df.rce, label=label, s=POINT_SIZE)
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.yscale('log')
    plt.ylim((10e-8, 10e+1))
    plt.ylabel("Reconstruction Error", fontsize=FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.legend(fontsize=18, loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(f"{config.outcome_dir}/scatter_plot.png")
    plt.clf()
