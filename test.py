import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=4, dilation=4, stride=2)

    def forward(self, x):
        return self.conv(x)


net = Net()
x = torch.randn(1, 3, 18, 18)
y = net(torch.autograd.Variable(x))
print(y.data.shape)
