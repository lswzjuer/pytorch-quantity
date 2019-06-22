from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import activation_quantizer
from model.ResNet import resnet50



    
def main():
    model = resnet50()
    # for name, m in model.named_modules():
    #     if type(m).__name__ in ['Conv2d', 'Linear', 'Eltwise', 'Concat', 'MaxPool2d', 'ReLU', 'UpsamplingNearest2d', 'myView']:
    #         print(name)
    model.eval()
    seed = 100
    torch.manual_seed(seed)
    inputs = torch.tensor(1, 3, 224, 224)
    out = model(inputs)
    
    np.save(out,'out.npy')

    test_lenet = activation_quantizer.Quantity(model)
    # test_lenet.activation_quantize(data_sets)
    #test_lenet.weight_quantize()
    #test_lenet.rewrite_weight()

if __name__ == "__main__":
    main()