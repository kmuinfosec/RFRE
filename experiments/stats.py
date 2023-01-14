import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from config import Config
from utils import get_time_window, TQDM


def save_stats(config: Config):
    key_rce_dict = {}
    for file_name in os.listdir(config.result_dir):
        result_csv_path = os.path.join(config.result_dir, file_name)
        test_df = pd.read_csv(result_csv_path)
        for row in test_df.itertuples():
            key = row.key
            rce = row.rce
            key_rce_dict[key] = rce

    score_df = []
    for test_dir in TQDM(config.test_dir_list, desc="loading results for statistics"):
        for file_name in os.listdir(test_dir):
            file_path = os.path.join(test_dir, file_name)
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

                score = key_rce_dict[key]
                label = row.Label
                score_df.append([key, score, label])
    score_df = pd.DataFrame(score_df, columns=['key', 'rce', 'label'])
    save_auc(score_df, config.outcome_dir)
    save_eval(score_df, config.outcome_dir)


def save_auc(test_df, outcome_dir):
    auc_csv = ['label,auc']
    test_df.loc[test_df['label'].str.startswith('BruteForce'), 'label'] = 'BruteForce-SSH'
    for label in test_df['label'].unique():
        if label.lower() == 'benign':
            continue
        benign_df = test_df[test_df['label'].str.lower() == 'benign']
        current_label_df = test_df[test_df['label'] == label]
        total_df = pd.concat((benign_df, current_label_df), ignore_index=True)
        total_df.loc[total_df['label'] == label, 'label'] = 1
        total_df.loc[total_df['label'].str.lower() == 'benign', 'label'] = 0
        fpr_list, tpr_list, threshold_list = roc_curve(total_df['label'].astype(int), total_df['rce'])

        auc_csv.append(f"{label},{auc(fpr_list, tpr_list)}")

    benign_df = test_df[test_df['label'].str.lower() == 'benign']
    attack_df = test_df[test_df['label'].str.lower() != 'benign']
    total_df = pd.concat((benign_df, attack_df), ignore_index=True)
    total_df.loc[total_df['label'].str.lower() != 'benign', 'label'] = 1
    total_df.loc[total_df['label'].str.lower() == 'benign', 'label'] = 0
    fpr_list, tpr_list, threshold_list = roc_curve(total_df['label'].astype(int), total_df['rce'])
    auc_csv.append(f"total,{auc(fpr_list, tpr_list)}")
    plot_roc_curve(fpr_list, tpr_list, outcome_dir)

    with open(os.path.join(outcome_dir, "auc.csv"), 'w', encoding='latin1') as f:
        f.write("\n".join(auc_csv))


def plot_roc_curve(fpr, tpr, outcome_dir):
    font_size = 30
    fig = plt.figure(figsize=(16, 9))
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)
    plt.title('Receiver Operating Characteristic Curve', fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.savefig(os.path.join(outcome_dir, 'auroc.jpg'), dpi=300)
    plt.clf()


def save_eval(test_df, outcome_dir):
    with open(os.path.join(outcome_dir, 'threshold.txt'), 'r') as f:
        threshold = float(f.readlines()[-1])

    detected_df = test_df[test_df['rce'] >= threshold]
    undetected_df = test_df[test_df['rce'] < threshold]

    tp = len(detected_df[detected_df['label'].str.lower() != 'benign'])
    fp = len(detected_df[detected_df['label'].str.lower() == 'benign'])

    tn = len(undetected_df[undetected_df['label'].str.lower() == 'benign'])
    fn = len(undetected_df[undetected_df['label'].str.lower() != 'benign'])

    acc = (tp + tn) / (tp + tn + fp + fn)
    pre = 0
    if tp + fp != 0:
        pre = tp / (tp + fp)
    rec = 0
    if tp + fn != 0:
        rec = tp / (tp + fn)
    f1 = 0
    if pre + rec != 0:
        f1 = 2 * (pre * rec) / (pre + rec)

    eval_csv = ['tp,fp,tn,fn,acc,pre,rec,f1']
    eval_csv.append(f"{tp},{fp},{tn},{fn},{acc},{pre},{rec},{f1}")

    with open(os.path.join(outcome_dir, 'eval.csv'), 'w') as f:
        f.write("\n".join(eval_csv))
