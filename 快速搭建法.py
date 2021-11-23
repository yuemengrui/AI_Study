"""
搭建神经网络的两种方法
net2会把激励函数一同纳入
net1中，激励函数实际上是在forword()功能中才被调用
相比于net2, net1可以根据个人需要更加个性化自己的前向传播过程，比如（RNN）
"""

import torch
import torch.nn.functional as F


# 搭建神经网络，方法一
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 隐藏层线性输出
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 输出层线性输出
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


net1 = Net(1, 10, 1)

print(net1)

# 搭建神经网络，方法二
net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

print(net2)
