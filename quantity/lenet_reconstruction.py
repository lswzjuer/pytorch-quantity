# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-06-17 11:14:11
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-06-21 17:55:15

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import reconstruction
from model.lenet.lenet import Cnn
from  torch.autograd import Variable
import torch
import torch.nn as nn
import os 
import yaml


def main():
    # load userconfigs files
    assert os.path.isfile("configs.yml"), "configs.yml is necessary"
    assert os.path.isfile("user_configs.yml"), "user_configs.yml is necessary"
    
    #read user config file
    with open("./user_configs.yml") as f:
        user_config = yaml.load(f)    

    data_path=user_config['PATH']['DATA_PATH']
    model_pth=user_config['PATH']['MODEL_PATH']
    quantity_model_path=user_config['PATH']['QUANTITY_MODEL_PATH']

    # load dataset
    batch_size=1
    test_dataset = datasets.MNIST(
        root=data_path, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # load model
    model = Cnn(1,10).eval()
    model.load_state_dict(torch.load(model_pth))
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    # test origin model
    criterion = nn.CrossEntropyLoss()
    eval_loss = 0
    eval_acc = 0

    for data in test_loader:
        image, label = data
        if use_gpu:
            img = Variable(image, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(image, volatile=True)
            label = Variable(label, volatile=True)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))

    # # Model reconstruction based on quantitative information
    # # the acticate/weight quantity must precede this
    # model_recon=reconstruction.Reconstruction(model)
    # all_quantity_info=model_recon.get_quantity_information()
    # quantity_model=model_recon.ReconModel(all_quantity_info,quantity_model_path)

    # # inference of quantity model
    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     quantity_model = quantity_model.cuda()
    # eval_loss = 0
    # eval_acc = 0
    # for data in test_loader:
    #     image, label = data
    #     if use_gpu:
    #         img = Variable(image, volatile=True).cuda()
    #         label = Variable(label, volatile=True).cuda()
    #     else:
    #         img = Variable(image, volatile=True)
    #         label = Variable(label, volatile=True)
    #     out = quantity_model(img)
    #     loss = criterion(out, label)
    #     eval_loss += loss.item() * label.size(0)
    #     _, pred = torch.max(out, 1)
    #     num_correct = (pred == label).sum()
    #     eval_acc += num_correct.item()
    # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    #     test_dataset)), eval_acc / (len(test_dataset))))


    # The test of the second equivalent quantization scheme
    model_recon=reconstruction.Reconstruction(model)
    all_quantity_info=model_recon.get_quantity_information()
    print(all_quantity_info)
    quantity_model=model_recon.ReconTest(all_quantity_info,quantity_model_path)
    print(quantity_model)

    print('image bit is :{}'.format(all_quantity_info['image']['output_bit']))

    # inference of quantity model
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        quantity_model = quantity_model.cuda()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        image, label = data
        # 量化反量化输入
        image=torch.div(torch.round(
                                torch.mul(image, 
                                                pow(2,all_quantity_info['image']['output_bit'])
                                          )
                                    ).clamp(-128.0, 127.0),
                        pow(2,all_quantity_info['image']['output_bit'])
                        )
        if use_gpu:
            img = Variable(image, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(image, volatile=True)
            label = Variable(label, volatile=True)
        out = quantity_model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset)))) 

if __name__ == "__main__":
    main()

