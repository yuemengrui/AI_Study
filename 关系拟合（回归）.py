"""
见证神经网络是如何通过简单的形式将一群数据用一条线条来表示
或者说
是如何在数据当中找到他们的关系，然后用神经网络模型来建立一个可以表示他们关系的线条
"""

import torch
import torch.nn.functional as F  # 激励函数
import matplotlib.pyplot as plt

# 建立数据集
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


# 建立神经网络（继承torch中的一个神经网络）
class Net(torch.nn.Module):    # 继承torch的Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 定义每层用什么样的形式
        # 隐藏层线性输出
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 输出层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # 这里同时也是Module中的forward功能
    def forward(self, x):
        # 正向传播输入值，神经网络分析出输出值
        # 激励函数（隐藏层的线性值）
        x = F.relu(self.hidden(x))
        # 输出值
        x = self.predict(x)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)

# 训练网络
# optimizer是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)   # 传入net的所有参数，学习率
# 预测值和真实值的误差计算公式（均方差）
loss_func = torch.nn.MSELoss()

plt.ion()

for t in range(200):
    # 给net训练数据x, 输出预测值
    prediction = net(x)

    # 计算两者的误差
    loss = loss_func(prediction, y)

    # 清空上一步的残余更新参数值
    optimizer.zero_grad()
    # 误差反向传输，计算参数更新值
    loss.backward()
    # 将参数更新值施加到net的parameters上
    optimizer.step()

    # 可视化训练过程
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

# 画图
plt.ioff()
plt.show()
