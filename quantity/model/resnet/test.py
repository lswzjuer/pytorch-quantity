import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torchvision.transforms as transforms
from ResNet import resnet50
# 定义超参数
batch_size = 64



def to_np(x):
    return x.cpu().data.numpy()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.CIFAR10(
    root='~/data/resnet_data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.CIFAR10(
    root='~/data/resnet_data', train=False, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = resnet50(pretrained=False )

# model = lenet.Cnn(1, 10)  # 图片大小是28x28
# model.load_state_dict(torch.load('lenet.pth'))

# use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
# if use_gpu:
#     model = model.cuda()


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    correct = 0
    total = 0
    for data in test_loader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
