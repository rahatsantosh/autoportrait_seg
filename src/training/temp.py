import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
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

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    return thresholded.mean()

img_path = "../../data/processed/CelebAMask-HQ/imgs"
mask_path = "../../data/processed/CelebAMask-HQ/mask"
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
])

train_data = Dataset(img_path, mask_path, transform, "train")
trainingloader = DataLoader(train_data, batch_size = 8)

valid_data = Dataset(img_path, mask_path, transform, "val")
validationloader = DataLoader(valid_data, batch_size = 8)

criterion = nn.MSELoss()

def test_model(model, train_loader, validation_loader, criterion, n_epochs, show_epoch=True, gpu=True):
    print("-------------------------------------------------")
    print("MODEL  : ", model.__class__.__name__)
    optimizer = torch.optim.Adam(model.parameters())
    accuracy_list=[]
    loss_list=[]
    for epoch in range(n_epochs):
        for x, y in train_loader:
            if gpu:
                # Transfer to GPU
                x, y = x.to(device), y.to(device)

            model.train()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data)

        if show_epoch:
            print("Epoch : ", epoch, "    Loss : ", loss.data)

        for x_test, y_test in validation_loader:
            if gpu:
                # Transfer to GPU
                x_test, y_test = x_test.to(device), y_test.to(device)
            model.eval()
            y_test = tf(y_test)
            z = model(x_test)
            iou = iou(z, y_test)
            accuracy_list.append(iou)
    print("-------------------------------------------------")
    return loss_list, accuracy_list

deeplab_loss, deeplab_acc = test_model(deeplab,
    trainingloader,
    validationloader,
    criterion,
    n_epochs = 100,
    show_epoch=True,
    gpu=use_cuda)

epoch = np.arrange(len(fcn_loss))

plt.plot(epoch, fcn_acc, label="FCN")
plt.plot(epoch, deeplab_acc, label="DeepLabv3")
plt.xlabel("Training Cycles")
plt.ylabel("IOU Metric")
plt.savefig('../../references/fig/iou.png')

def pickle_dump(path, list):
    with open(path, "wb") as fp:
        pickle.dump(list, fp)

pickle_dump("../../reports/performance_data/fcn_loss.txt", fcn_loss)
pickle_dump("../../reports/performance_data/fcn_acc.txt", fcn_acc)
pickle_dump("../../reports/performance_data/deeplab_loss.txt", deeplab_loss)
pickle_dump("../../reports/performance_data/deeplab_acc.txt", deeplab_acc)

'''
Function to calculate iou metric, to get an accuracy of predicted mask.
Code adapted from :
https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
'''
