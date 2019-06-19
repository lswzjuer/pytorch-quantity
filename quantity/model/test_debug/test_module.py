import torch
import torch.nn as nn
from torchsummary import summary
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        submodule = nn.Conv2d(10, 10, 4)
        self.conv0 = nn.Conv2d(10, 10, 4)
        #self.conv = nn.Conv2d(10, 20, 4)
        #self.conv1 = nn.Conv2d(10, 20, 4)
        self.add_module("conv", submodule)
        self.add_module("conv1",submodule)
    def forward(self, input):
        x = self.conv0(input)
        x = self.conv(x)
        x = self.conv1(x)
        return x


