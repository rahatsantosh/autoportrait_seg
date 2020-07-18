import os
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision
from dataset import Dataset
from models import Autoencoder1

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True
print(torch.cuda.get_device_name())

img_path = "../../data/processed/CelebAMask-HQ/imgs"
mask_path = "../../data/processed/CelebAMask-HQ/mask"

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomRotation(45),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = 0.0, std = 1.0, inplace=True)
])

train_data = Dataset(img_path, mask_path, transform, "train")
trainloader = DataLoader(train_data, batch_size = 6)
valid_data = Dataset(img_path, mask_path, transform, "val")
validationloader = DataLoader(valid_data, batch_size = 6)

def train_model(model,train_loader,validation_loader,optimizer,n_epochs=4,gpu=True):
    #accuracy_list=[]
    loss_list=[]
    tf = torchvision.transforms.Normalize(mean = 0.0, std = 1.0, inplace=True)
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
        print("Epoch : ", epoch, "    Loss : ", loss.data)

        '''for x_test, y_test in validation_loader:
            if gpu:
                # Transfer to GPU
                x_test, y_test = x_test.to(device), y_test.to(device)
            model.eval()
            y_test = tf(y_test)
            z = model(x_test)
            iou = mean_iou(z, y_test, 1)
            accuracy_list.append(accuracy)'''

    return loss_list

model = Autoencoder1()
if use_cuda:
	model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

loss_list = train_model(
    model=model,
    n_epochs=2,
    train_loader=trainloader,
    validation_loader=validationloader,
    optimizer=optimizer,
    gpu=True)

'''plt.plot(np.arange(len(accuracy_list)),accuracy_list)
plt.show()'''

plt.plot(np.arange(len(loss_list)),loss_list)
plt.show()
