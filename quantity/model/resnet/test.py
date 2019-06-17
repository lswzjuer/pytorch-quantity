from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
from ResNet import resnet50
import torch



    
def main():
    model = resnet50()
    # for name, m in model.named_modules():
    #     if type(m).__name__ in ['Conv2d', 'Linear', 'Eltwise', 'Concat', 'MaxPool2d', 'ReLU', 'UpsamplingNearest2d', 'myView']:
    #         print(name)
    model.load_state_dict(torch.load('resnet50.pth'))
    model.eval()
    seed = 100
    torch.manual_seed(seed)
    inputs = torch.rand(1, 3, 224, 224)
    out = model(inputs)
    np.save('out.npy',out.detach().numpy())

    #test_lenet.rewrite_weight()

if __name__ == "__main__":
    main()