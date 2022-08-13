import os
import sys
import numpy as np
import torch
from tqdm import tqdm

from ml.model import AutoEncoder
from utils import get_device


def trainer(dataloader, epochs, outcome_dir_path):
    model = AutoEncoder(21, 128, 8, device=get_device())
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    pbar = tqdm(range(epochs), ascii=True, file=sys.stdout, desc="training")
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
            pbar.set_postfix(epoch=f"{e + 1} of {epochs}", loss=f"{loss_per_batch[-1]:.5f}")
        loss_history.append(np.mean(loss_per_batch))

    model_path = os.path.join(outcome_dir_path, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model, f)

    with open(os.path.join(outcome_dir_path, 'model.txt'), 'w') as f:
        f.write(str(model) + '\n')

    with torch.inference_mode():
        model.eval()
        evaluation_rce_list = []
        for batch, _ in tqdm(dataloader, desc='evaluating train', file=sys.stdout):
            recons = model(batch)
            rces = model.mse(batch, recons, reduction='none')
            evaluation_rce_list += rces.tolist()

    threshold = np.percentile(sorted(evaluation_rce_list), 100)

    with open(os.path.join(outcome_dir_path, 'threshold.txt'), 'w') as f:
        f.write(str(threshold)+"\n")
