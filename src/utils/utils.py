import numpy as np
import cv2
from PIL import Image
import torch
import numpy as np
from PIL import Image
import torchvision
from torch.backends import cudnn
from visualization_utils import *

def get_mask(img, model):
	#CUDA for PyTorch
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	cudnn.benchmark = True

	model.to(device)
	model.eval()

	tf = torchvision.transforms.Compose([
	    torchvision.transforms.ToTensor(),
	    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
	])
	img = tf(img).to(device)

	mask = model(img)

	return infer_op(mask)

def predict_mask(img_path, model):
	img = Image.open(image_path)
	ip_size = (224, 224)
	img = img.resize(ip_size))

	mask = get_mask(img, model)

	return mask
