"""
用最简单的途径看神经网络是怎么进行事物的分类
"""

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

# x, y = Variable(x), Variable(y)
#
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 隐藏层线性输出
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 输出层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 正向传播输入值，神经网络分析出输出值
        # 激励函数（隐藏层的线性值）
        x = F.relu(self.hidden(x))
        # 输出值，非预测值，预测值还需另外计算
        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)
# net的结构
print(net)

# 训练网络
# optimizer是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()

for t in range(100):
    out = net(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 可视化训练
    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
