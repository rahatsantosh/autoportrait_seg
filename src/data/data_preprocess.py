import glob
import shutil
import os
import numpy as np
import cv2
import split_folders

path = "../../data/processed/CelebAMask-HQ"
os.mkdir(path)
os.mkdir(os.path.join(path, "mask_"))
os.mkdir(os.path.join(path, "mask"))
os.mkdir(os.path.join(path, "img"))
os.mkdir(os.path.join(path, "img/img"))

destination_path = "../../data/processed/CelebAMask-HQ/mask_/"
pattern = "../../data/raw/CelebAMask-HQ/CelebAMask-HQ-mask-anno/*/*"
for img in glob.glob(pattern):
    shutil.copy(img, destination_path)
print("Transferred")

ann_path = "../../data/processed/CelebAMask-HQ/mask/"
categories = [
    'hair.png',
    'l_brow.png',
    'r_brow.png',
    'l_eye.png',
    'r_eye.png',
    'nose.png',
    'l_lip.png',
    'u_lip.png',
    'neck.png',
    'skin.png',
    'mouth.png',
    'l_ear.png',
    'r_ear.png',
    'cloth.png'
]

for i in range(30000):
    path = destination_path + "{:05d}".format(i) + "_"
    combined_ann = np.zeros((512, 512, 3)).astype(np.uint8)
    for cat in categories:
        path_cat = path + cat
        if not os.path.exists(path_cat):
            continue
        img = cv2.imread(path_cat)
        combined_ann = cv2.add(combined_ann, img)
    write_path = ann_path + str(i) + ".png"
    cv2.imwrite(write_path, combined_ann)

print("Annotations combined")

shutil.rmtree(destination_path)
print("mask_ directory removed")

img_path = "../../data/raw/CelebAMask-HQ/CelebA-HQ-img"
img_dest = "../../data/processed/CelebAMask-HQ/img/img"

for f in os.listdir(img_path):
    path = os.path.join(img_path, f)
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 512))
    dest = os.path.join(img_dest, f)
    cv2.imwrite(dest, img)

print("Images resized and transferred")

target_path = "../../data/processed/CelebAMask-HQ/img"
output_path = "../../data/processed/CelebAMask-HQ/imgs"
os.mkdir(output_path)

split_folders.ratio(target_path, output=output_path, seed=23, ratio=(.8, .1, .1))

print("Test/Train/Val split")
shutil.rmtree(target_path)
