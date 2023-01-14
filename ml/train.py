import sys
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import Config
from ml.model import AutoEncoder
from utils import get_device, TQDM


def trainer(config: Config, dataloader: DataLoader):
    model = AutoEncoder(21, 128, 8, device=get_device())
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())

    pbar = tqdm(range(config.epochs), ascii=True, file=sys.stdout, desc="training")
    loss_history = []
    for e in pbar:
        loss_per_batch = []
        for batch, _ in dataloader:
            recons = model(batch)
            rce = model.mse(batch, recons)
            optimizer.zero_grad()
            rce.backward()
            optimizer.step()
            loss_per_batch.append(rce.item())
            pbar.set_postfix(epoch=f"{e + 1} of {config.epochs}", loss=f"{loss_per_batch[-1]:.5f}")
        loss_history.append(np.mean(loss_per_batch))

    with open(config.model_path, 'wb') as f:
        torch.save(model, f)

    with open(config.model_txt_path, 'w') as f:
        f.write(str(model) + '\n')

    with torch.inference_mode():
        model.eval()
        evaluation_rce_list = []
        for batch, _ in TQDM(dataloader, desc='evaluating train'):
            recons = model(batch)
            rces = model.mse(batch, recons, reduction='none')
            evaluation_rce_list += rces.tolist()

    threshold = np.percentile(sorted(evaluation_rce_list), 100)

    with open(config.threshold_path, 'w') as f:
        f.write(str(threshold)+"\n")


def tester(model, dataloader):
    result_csv = ['key,rce']
    with torch.inference_mode():
        for batch, group_key_batch in TQDM(dataloader, desc="testing"):
            recons = model(batch)
            rces = model.mse(batch, recons, reduction='none')
            for i in range(len(rces)):
                rce = rces[i].item()
                key = group_key_batch[i]
                result_csv.append(",".join([key, str(rce)]))

    return result_csv
