import csv
import copy
import time
from tqdm import tqdm
import torch
import numpy as np
import os


def model_validation(model, dataloaders, metrics):
    since = time.time()
    # CUDA for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize batch summary
    batchsummary = {keys: [0] for keys in metrics.keys()}

    model.eval()   # Set model to evaluate mode

    # Iterate over data.
    for inputs, masks in tqdm(iter(dataloaders[phase])):
        inputs = inputs.to(device)
        masks = masks.to(device)

        outputs = model(inputs)
        y_pred = outputs['out'].data.cpu().numpy().ravel()
        y_true = masks.data.cpu().numpy().ravel()
        for name, metric in metrics.items():
            if name == 'f1_score':
                # Use a classification threshold of 0.1
                batchsummary[name].append(
                    metric(y_true > 0, y_pred > 0.1))
            else:
                batchsummary[name].append(
                    metric(y_true.astype('uint8'), y_pred))


    for field in fieldnames.keys():
        batchsummary[field] = np.mean(batchsummary[field])

    time_elapsed = time.time() - since
    print('Validation time in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return batchsummary, time_elapsed
