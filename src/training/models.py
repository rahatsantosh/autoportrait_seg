import torch
from torch import nn
import torchvision

deeplab = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
for param in deeplab.parameters():
    param.requires_grad = False

fcn = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
for param in fcn.parameters():
    param.requires_grad = False

class Mask_deeplab(nn.Sequential):
  def __init__(self):
    super(Mask_deeplab, self).__init__(
        nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
    )

class Mask_fcn(nn.Sequential):
  def __init__(self):
    super(Mask_fcn, self).__init__(
        nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))
    )

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


def refit_model(model, Mask_module):
    model.aux_classifier = Identity()
    model.classifier[4] = Mask_module()

    for param in model.classifier[1].parameters():
      param.requires_grad = True
    for param in model.classifier[2].parameters():
      param.requires_grad = True
    return model

fcn = refit_model(fcn, Mask_fcn)
deeplab = refit_model(deeplab, Mask_deeplab)

fcn_path = "../../models/fcn.pt"
deeplab_path = "../../models/deeplab.pt"

torch.save(fcn, fcn_path)
torch.save(deeplab, deeplab_path)
