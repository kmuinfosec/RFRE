import os
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Dict

from config import Config
from utils import get_time_window, TQDM


def save_stats(config: Config):
    profile_rce_dict = {}
    for file_name in os.listdir(config.path.result_dir):
        result_csv_path = os.path.join(config.path.result_dir, file_name)
        with open(result_csv_path, 'r') as f:
            lines = f.readlines()[1:]

        for line in lines:
            profile_key, rce = line.split(',')
            profile_rce_dict[profile_key] = float(rce)

    seq_num = 0
    label_score_dict = {}
    for test_dir in TQDM(config.test_dir_list, desc="loading results for statistics"):
        for csv_name in os.listdir(test_dir):
            csv_path = os.path.join(test_dir, csv_name)
            with open(csv_path, 'r') as f:
                flows = f.readlines()[1:]

            for flow in flows:
                flow = flow.split(',')
                sip, dip = flow[config.column_index['sip']], flow[config.column_index['dip']]
                ws, we = get_time_window(flow[config.column_index['time_start']], config.time_window)
                label = flow[config.column_index['label']].strip()
                if label.lower().strip() == 'benign':
                    label = 'benign'

                if sip in config.inside_ip_set:
                    profile_key = f'{dip}_{ws}_{we}'
                elif dip in config.inside_ip_set:
                    profile_key = f'{sip}_{ws}_{we}'
                else:
                    continue
                score = profile_rce_dict[profile_key]

                if label not in label_score_dict:
                    label_score_dict[label] = {'score_list': [], 'seq_num_list': []}
                label_score_dict[label]['seq_num_list'].append(seq_num)
                label_score_dict[label]['score_list'].append(score)
                seq_num += 1

    save_scatter_plot(label_score_dict, config.path.threshold_path, config.path.scatter_plot_path)
    save_auc(label_score_dict, config.path.auroc_path, config.path.roc_curve_path)
    save_eval(label_score_dict, config.path.threshold_path, config.path.eval_path)


def save_auc(label_score_dict: Dict[str, Dict[str, list]], auroc_path, roc_curve_path):
    auc_csv = ['label,auc']
    benign_scores = label_score_dict['benign']['score_list']
    benign_labels = ['benign' for _ in range(len(benign_scores))]

    total_scores = [] + benign_scores
    total_labels = [] + benign_labels
    for label, items in label_score_dict.items():
        if label == 'benign':
            continue
        labels = ['positive' for _ in range(len(items['score_list']))]
        scores = items['score_list']
        total_scores += scores
        total_labels += labels
        fpr_list, tpr_list, threshold_list = roc_curve(benign_labels+labels, benign_scores+scores, pos_label='positive')
        auc_csv.append(f"{label},{auc(fpr_list, tpr_list)}")

    fpr_list, tpr_list, threshold_list = roc_curve(total_labels, total_scores, pos_label='positive')
    auc_csv.append(f"total,{auc(fpr_list, tpr_list)}")
    with open(auroc_path, 'w', encoding='latin1') as f:
        f.write("\n".join(auc_csv))

    plotting_roc_curve(fpr_list, tpr_list, roc_curve_path)


def save_eval(label_score_dict: Dict[str, Dict[str, list]], threshold_path: str, eval_path: str):
    with open(threshold_path, 'r') as f:
        threshold = float(f.readlines()[-1])

    tp, fp, tn, fn = 0, 0, 0, 0
    for label, items in label_score_dict.items():
        for score in items['score_list']:
            if label == 'benign':
                if score >= threshold:
                    fp += 1
                else:
                    tn += 1
            else:
                if score >= threshold:
                    tp += 1
                else:
                    fn += 1

    acc = (tp + tn) / (tp + tn + fp + fn)
    pre = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * (pre * rec) / (pre + rec) if pre + rec else 0

    eval_csv = f'tp,fp,tn,fn,acc,pre,rec,f1\n{tp},{fp},{tn},{fn},{acc},{pre},{rec},{f1}\n'
    with open(eval_path, 'w') as f:
        f.write(eval_csv)


def plotting_roc_curve(fpr, tpr, roc_curve_path):
    font_size = 30
    plt.figure(figsize=(16, 9))
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)
    plt.title('Receiver Operating Characteristic Curve', fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.savefig(roc_curve_path, dpi=300)
    plt.clf()


def save_scatter_plot(label_score_dict: Dict[str, Dict[str, list]], threshold_path: str, scatter_plot_path: str):
    font_size = 30
    legend_font_size = 15
    point_size = 90
    num_colors = 20
    cm = plt.get_cmap('tab20')

    with open(threshold_path, 'r') as f:
        threshold = float(f.readlines()[-1].strip())

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

    ax.scatter(label_score_dict['benign']['seq_num_list'], label_score_dict['benign']['score_list'], label='benign', alpha=0.25)
    for label, items in label_score_dict.items():
        if label == 'benign':
            continue
        plt.scatter(items['seq_num_list'], items['score_list'], label=label, s=point_size)
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.yscale('log')
    plt.ylim((10e-7, 10e+1))
    plt.ylabel("Reconstruction Error", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(fontsize=legend_font_size, loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(scatter_plot_path)
    plt.clf()
