import torch
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLoss(nn.Module):
	def __init__(self):
		super(MyLoss, self).__init__()
		return 

	def forward(self, pred, true):
		loss = F.cross_entropy(pred, true)
		loss = torch.mean(loss)
		return loss
		