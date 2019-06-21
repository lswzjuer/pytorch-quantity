# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-06-17 11:14:11
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-06-21 18:45:59

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import reconstruction
import model.resnet.ResNet_18_fabu as resnet
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


    batch_size = 100
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test)
    testloader =DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # load model
    model = resnet.ResNet18()
    model.load_state_dict(torch.load(model_pth,map_location='cpu'))
    # print model information to examine
    print(model)
    state_dict=model.state_dict()
    for name,value in state_dict.items():
        print("name is : {} size is : {}".format(name,value.size()))

    ################ test origin model################################
    model_test=model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        i=0
        for data in testloader:
            i+=100
            if i>=500:
                break
            images, labels = data
            outputs = model_test(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('origin model acc is :%.3f' % (100 * correct / total))

    ###################merge bn layer##################################
    model_recon=reconstruction.Reconstruction(model)
    merge_bn_model=model_recon.merge_bn().eval()
    # print model information to examine
    print(merge_bn_model)
    state_dict=merge_bn_model.state_dict()
    for name,value in state_dict.items():
        print("name is : {} size is : {}".format(name,value.size()))

    # test the merge_bn model
    with torch.no_grad():
        correct = 0
        total = 0
        i=0
        for data in testloader:
            i+=100
            if i>=500:
                break
            images, labels = data
            outputs = merge_bn_model(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('merge bn model acc is :%.3f' % (100 * correct / total))

    # Because of the model Shared memory, the merge_bn function has changed the original model
    # They are exactly the same
    print(model)
    print(merge_bn_model)


    #################量化->卷积->右移->BiasAdd->饱和截断->反量化########################
    all_quantity_info=model_recon.get_quantity_information()
    quantity_model=model_recon.ReconModel(all_quantity_info,quantity_model_path)
    with torch.no_grad():
        correct = 0
        total = 0
        i=0
        for data in testloader:
            i+=100
            if i>=500:
                break
            images, labels = data
            outputs = quantity_model(images)
            _, predicted = torch.max(outputs, 1)
            print("size is : {} : {}".format(predicted.size(),labels.size()))
            total += labels.size(0)
            correct += (predicted == labels).sum()
            print("the sample has been inferenced : {}".format(i))
        print('reconstruction model acc is : %.3f' % (100 * correct / total))

    # #################w,b,output quantity and dequantity test########################
    # all_quantity_info=model_recon.get_quantity_information()
    # quantity_model=model_recon.ReconTest(all_quantity_info,quantity_model_path)
    # print(quantity_model)
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     i=0
    #     for data in testloader:
    #         i+=100
    #         if i>=500:
    #             break
    #         images, labels = data
    #         images=torch.div(torch.round(torch.mul(images, pow(2, 5))).clamp(-128.0, 127.0), pow(2, 5))
    #         outputs = quantity_model(images)
    #         _, predicted = torch.max(outputs, 1)
    #         print("size is : {} : {}".format(predicted.size(),labels.size()))
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum()
    #         print("the sample has been inferenced : {}".format(i))
    #     print('q-dq reconstruction model acc is : %.3f' % (100 * correct / total))

if __name__ == "__main__":
    main()



