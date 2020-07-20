import torch
from torch import nn
import torchvision

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

deeplab = torchvision.models.segmentation.deeplabv3_resnet101(num_classes=1, aux_loss=True)
deeplab_dict = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).state_dict()
for param in deeplab.parameters():
    param.requires_grad = False

def repurpose(model_dict, fin_channels):
    model_dict['aux_classifier.4.weight'] = torch.rand([1, fin_channels, 1, 1], dtype=torch.float32,device=device,requires_grad=True)
    model_dict['aux_classifier.4.bias'] = torch.rand([1], dtype=torch.float32,device=device,requires_grad=True)
    model_dict['classifier.4.weight'] = torch.rand([1, fin_channels, 1, 1], dtype=torch.float32,device=device,requires_grad=True)
    model_dict['classifier.4.bias'] = torch.rand([1], dtype=torch.float32,device=device,requires_grad=True)

    return model_dict

def finetune_prep(model):
    for param in model.classifier[4].parameters():
      param.requires_grad = True
    for param in model.aux_classifier[4].parameters():
      param.requires_grad = True
    return model

deeplab.load_state_dict(repurpose(deeplab_dict, 256))
deeplab = finetune_prep(deeplab)

deeplab_path = "../../models/deeplab.pt"
torch.save(deeplab, deeplab_path)
