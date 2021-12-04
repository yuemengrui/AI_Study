import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
from loss import MyLoss

from torch.autograd import Variable
from dataset import ListDataset
from model import VGG

trainset = ListDataset(file_path='./data/small_train/', training=True)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)

testset = ListDataset(file_path='./data/test/', training=False)
testLoader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=8)

model = VGG()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train(model, trainLoader, optimizer, loss_func):
	model.train()
	all_loss = 0
	for batch_id, (img, label) in enumerate(trainLoader):
		if torch.cuda.is_available():
			img = img.cuda()
			label = label.cuda()
		img = Variable(img)
		label = Variable(label)

		optimizer.zero_grad()
		pred = model(img)
		loss = loss_func(pred, label)
		all_loss += loss.data.item()
		loss.backward()
		optimizer.step()
		if batch_id % 50 == 0:
			print('Train set: Step: {}, loss: {:.4f}'.format(batch_id, loss.data.item()))

	avg_loss = all_loss/len(trainLoader)
	print("Train Avg Loss : {:.6f}".format(avg_loss))

def test(model, testLoader, loss_func):
	model.eval()
	all_loss = 0
	correct = 0
	for batch_idx, (img, label) in enumerate(testLoader):
		if torch.cuda.is_available():
			img = img.cuda()
			label = label.cuda()
		img = Variable(img)
		label = Variable(label)

		optimizer.zero_grad()
		pred = model(img)
		loss = loss_func(pred, label)
		all_loss += loss.data.item()
		pred = pred.data.max(1)[1]
		correct += pred.eq(label.data).cpu().sum()
		# print(correct)

	avg_loss = all_loss/len(testLoader)
	accuracy = 100. * correct / len(testLoader.dataset)
	print("Test Avg Loss : {:.6f}, Accuracy:{:.2f}".format(avg_loss, accuracy))

if torch.cuda.is_available():
	print("22")
	model.cuda()
	model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
	cudnn.benchmark = True

#model.load_state_dict(torch.load("./checkpoint.pth"))

loss_func = MyLoss()
for epoch in range(1, 100):
	train(model=model, trainLoader=trainLoader, optimizer=optimizer, loss_func=loss_func)
	test(model, testLoader, loss_func)
	torch.save(model.state_dict(), './checkpoint1.pth')
