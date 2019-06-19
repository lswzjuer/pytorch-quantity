from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import activation_quantizer
from model.resnet.ResNet_18_fabu import ResNet18
from common.quantity.utils import merge_bn
import torch

import torchvision

    
def main():
    model = ResNet18()
    # for name, m in model.named_modules():
    #     if type(m).__name__ in ['Conv2d', 'Linear', 'Eltwise', 'Concat', 'MaxPool2d', 'ReLU', 'UpsamplingNearest2d', 'myView']:
    #         print(name)
    model_path = './model/resnet/resnet18.pth'
    model.load_state_dict(torch.load(model_path))
    #model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.cuda()
    model.eval()
   
    model = merge_bn(model)

    seed = 100
    torch.manual_seed(seed)
    inputs = torch.rand(1, 3, 32, 32)
    # out = model(inputs)
    # out_np = out.detach().numpy()
    # np.save('out.npy', out_np)


    # for name, param in model.named_parameters():
    #     print(name)
    # print('[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[')
    # for name, module in model.named_modules():
    #     if(type(module).__name__ in ['Conv2d', 'Linear', 'Eltwise']):
    #         print(name)

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
    test_lenet = activation_quantizer.Quantity(model)
    test_lenet.activation_quantize(testloader)
    test_lenet.weight_quantize()
    test_lenet.rewrite_weight()

if __name__ == "__main__":
    main()
