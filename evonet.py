# code modified from https://github.com/pytorch/examples/blob/main/mnist/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EvoNet(nn.Module):
    def __init__(self):
        super(EvoNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.init_w()


    def init_w(self):
        gain = nn.init.calculate_gain('relu')
        for m in self.children():
            nn.init.xavier_uniform_(m.weight, gain)


    def forward(self, x):
        # import pdb; pdb.set_trace()
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@torch.no_grad()
def evo_w(child, parent, beta):
    gain = nn.init.calculate_gain('relu')
    gain *= beta
    for m, pm in zip(child.children(), parent.children()):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        torch.nn.init._no_grad_uniform_(m.weight, -a, a)
        m.weight += pm.weight
