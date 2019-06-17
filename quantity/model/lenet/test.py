import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import lenet 
# 定义超参数
batch_size = 1



def to_np(x):
    return x.cpu().data.numpy()


# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = lenet.Cnn(1, 10)  # 图片大小是28x28
model.load_state_dict(torch.load('lenet.pth'))

use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()


# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
model.eval()
eval_loss = 0
eval_acc = 0

for data in test_loader:
    img, label = data
    if use_gpu:
        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
    else:
        img = Variable(img, volatile=True)
        label = Variable(label, volatile=True)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc / (len(test_dataset))))
