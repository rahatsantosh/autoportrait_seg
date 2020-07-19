from trainer import train_model
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.backends import cudnn
import models
from dataset import Dataset

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

deeplab = torch.load(models.deeplab_path)
deeplab.to(device)

train_data = Dataset(img_path, mask_path, transform, "train")
trainingloader = DataLoader(train_data, batch_size = 8)

valid_data = Dataset(img_path, mask_path, transform, "val")
validationloader = DataLoader(valid_data, batch_size = 8)

dataloaders = {"Train" : trainingloader, "Test" : validationloader}

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(deeplab.parameters())
metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
bpath = "../../reports/performance_data"

deeplab = train_model(deeplab, criterion, dataloaders, optimizer, metrics, bpath, num_epochs=30)
