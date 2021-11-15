# *_*coding:utf-8 *_*
# Author: ai_YF
# Time: 2021/11/15 21:28
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32* 7 *7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


transform = transforms.Compose([transforms.ToTensor()])


train_data = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=False)

test_data = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)

train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=64,
                                                shuffle=True)

test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=64,
                                               shuffle=True)

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(500):
    running_loss = 0.0
    running_correct = 0
    print("-"*10)
    print("Epoch {}/{}".format(epoch, 500))
    for data in train_data_loader:
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = cnn(X_train)
        _, pred = torch.max(outputs.data, 1)

        loss = loss_func(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)

    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%".format(running_loss/len(train_data),100*running_correct/len(train_data)))

    testing_correct = 0
    for data in test_data_loader:
        X_test, y_test = data
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = cnn(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data).item()
    print("Test Accuracy is:{:.4f}%".format(100*testing_correct/len(test_data)))

    if 100*testing_correct/len(test_data) > best_acc:
        best_acc = 100*testing_correct/len(test_data)
        torch.save(cnn.state_dict(), "./checkpoints/mnist_{}.pth".format(str(best_acc)))


