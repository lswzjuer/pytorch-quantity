from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import sys
sys.path.insert(0, '../')

from tools import Quantity
from model.resnet.ResNet_18_fabu import ResNet18
from common.quantity import merge_bn
import torch
import yaml

import torchvision
    
def main():
    model = ResNet18()

    model_path = '../model/resnet/resnet18.pth'

    #read user config file
    with open("./user_configs.yml") as f:
        user_config = yaml.load(f)
    device = user_config['SETTINGS']['DEVICE']
    if (device == 'gpu'):
            
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        model.eval()
        model = merge_bn(model, 'cuda')
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        model = merge_bn(model, 'cpu')
   
   

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='/private/zhangjiwei/data/CIFAR-10_DATA', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=True, num_workers=2)
    # Cifar-10的标签
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    test_lenet = Quantity(model)
    test_lenet.activation_quantize(testloader)
    test_lenet.weight_quantize()
    test_lenet.rewrite_weight()

if __name__ == "__main__":
    main()
