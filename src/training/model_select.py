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

checkpoint_path = "../../models/checkpoints"

deeplab = torch.load(models.deeplab_path)
deeplab.to(device)

img_path = "../../data/processed/CelebAMask-HQ/imgs"
mask_path = "../../data/processed/CelebAMask-HQ/mask"
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
])

transform_op = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

train_data = Dataset(img_path, mask_path, transform, transform_op, "train")
trainingloader = DataLoader(train_data, batch_size = 32)

valid_data = Dataset(img_path, mask_path, transform, transform_op, "val")
validationloader = DataLoader(valid_data, batch_size = 32)

dataloaders = {"Train" : trainingloader, "Test" : validationloader}

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(deeplab.parameters())
metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
bpath = "../../reports/performance_data"

deeplab = train_model(deeplab, criterion, dataloaders, optimizer, metrics, checkpoint_path, bpath, num_epochs=30)

torch.save(deeplab, models.deeplab_path)
