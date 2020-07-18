import os
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
class Dataset(Dataset):
   def __init__(self, img_path, mask_path, transform, split="train"):
       #assert split == ("train" or "test" or "val")

       self.img_list = []
       self.mask_list = []
       self.length = 0
       self.transform = transform
       img_path = os.path.join(img_path, split)
       img_path = os.path.join(img_path, "img")
       for f in os.listdir(img_path):
           d = f[:-3] + "png"
           self.img_list.append(os.path.join(img_path, f))
           self.mask_list.append(os.path.join(mask_path, d))
           self.length += 1

   def __len__(self):
       return self.length

   def __getitem__(self, index):

       seed = np.random.randint(2147483647)
       random.seed(seed)
       x = Image.open(self.img_list[index], 'r')
       x = self.transform(x)
       y = Image.open(self.mask_list[index], 'r')
       y = self.transform(y)

       return x, y
