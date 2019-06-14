from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import activation_quantizer
from model.ResNet import resnet50


    
def main():
    model = resnet50()
    for name, m in model.named_modules():
        if type(m).__name__ in ['Conv2d', 'Linear', 'Eltwise', 'Concat', 'MaxPool2d', 'ReLU', 'UpsamplingNearest2d', 'myView']:
            print(name)
    test_lenet = activation_quantizer.Quantity(model)
    # test_lenet.activation_quantize(data_sets)
    #test_lenet.weight_quantize()
    #test_lenet.rewrite_weight()

if __name__ == "__main__":
    main()