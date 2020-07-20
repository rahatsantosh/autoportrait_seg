import numpy as np
import cv2

def infer_op(op):
	op = op.cpu().numpy()
	op = op * 255
	op.astype(np.uint8)

	return op

def apply_mask(img, mask):
	assert np.shape(mask) == (224, 224)

	img = cv2.resize(img, (224, 224))
	portrait = cv2.bitwise_and(img, img, mask = mask)

	return portrait
