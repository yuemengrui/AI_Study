import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 建立数据集
# 数据的基本形态
n_data = torch.ones(100, 2)
# 类型0 x data (tensor), shape=(100, 2)
x0 = torch.normal(2 * n_data, 1)
# 类型0 y data (tensor), shape=(100, )
y0 = torch.zeros(100)
# 类型1 x data (tensor), shape=(100, 1)
x1 = torch.normal(-2 * n_data, 1)
# 类型0 y data (tensor), shape=(100, )
y1 = torch.ones(100)

# torch.cat是在合并数据
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, ):
        pass


