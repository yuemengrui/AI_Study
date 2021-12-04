import os
import torch
import torch.utils.data as data
import numpy as np
import random
import cv2
import torchvision.transforms as transforms
import re
from PIL import Image


class ListDataset(data.Dataset):
	def __init__(self, file_path, input_size=(224, 224), training = True):
		self.input_size = input_size
		self.training = training

		self.image_names = []
		self.labels = []
		self.dict = ["cats", "dogs"]
		for root, _, file_list in os.walk(file_path):
			label_name = os.path.basename(root)
			for file in file_list:
				self.image_names.append(os.path.join(root, file))
				self.labels.append(label_name)

	def __getitem__(self, idx):
		label_name = self.labels[idx]
		image_name = self.image_names[idx]
		img, label = get_label(image_name, label_name, self.dict)
		img = transform(img, self.input_size, self.training)
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		img = transforms.Compose([
			transforms.ToTensor(),
			normalize
		])(img)
		return img, label

	def __len__(self):
		return len(self.labels)

def transform(img, input_size, training=False):
	if training:
		if np.random.rand() < 0.5:
			angle = (np.random.rand() - 0.5) * 40
			img = rotate_bound(img, angle)

		if np.random.rand() < 0.5:
			img = cv2.flip(img, 1)

	img = cv2.resize(img, dsize=input_size)
	return img

def get_label(image_name, label_name, dictionary):
	image = cv2.imread(image_name)
	label = np.array(dictionary.index(label_name))
	return image, label

def rotate_bound(image, angle):
	(h, w) = image.shape[:2]
	(cX, cY) = (w / 2, h / 2)

	M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))

	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	return cv2.warpAffine(image, M, (nW, nH))

def test():
	trainset = ListDataset(file_path='./data/small_train/')
	testloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=1)

	for batch_idx, (img, label) in enumerate(testloader):
		# if batch_idx > 10 :
		# 	break
		continue

if __name__ == "__main__":
	test()
