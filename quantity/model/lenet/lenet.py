import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import os,sys

sys.path.insert(0, '../../')

from common.quantity import View

# 定义 Convolution Network 模型
class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(False),
            nn.MaxPool2d(2, 2))
        self.review = View()
        self.fc = nn.Sequential(
            nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = self.review(out)
        out = self.fc(out)
        return out
