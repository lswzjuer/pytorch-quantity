from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import activation_quantizer
from model.test_debug.test_module import Model


    
def main():
    model = Model()
    test_lenet = activation_quantizer.Quantity(model)
    # test_lenet.activation_quantize(data_sets)
    #test_lenet.weight_quantize()
    #test_lenet.rewrite_weight()

if __name__ == "__main__":
    main()
