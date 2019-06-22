import torch
import torch.nn as nn


class Eltwise(nn.Module):

    def __init__(self):
        super(Eltwise, self).__init__()

    def forward(self, x, y):
        return x + y


class Concat(nn.Module):

    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, x, y, dim=1):
        return torch.cat((x, y), dim=dim)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1).clone()
