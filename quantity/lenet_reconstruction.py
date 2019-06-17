# -*- coding: utf-8 -*-
# @Author: liusongwei
# @Date:   2019-06-17 11:14:11
# @Last Modified by:   liusongwei
# @Last Modified time: 2019-06-17 13:26:32

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import reconstruction
from model.lenet import Cnn
from  torch.autograd import Variable
import torch
import torch.nn as nn




def main():
    # load dataset
    batch_size=1
    test_dataset = datasets.MNIST(
        root='../../data', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # load model
    model = Cnn(1,10).eval()
    pth_path = './model/cnn.pth'  #weight path (.pth)
    model.load_state_dict(torch.load(pth_path))
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
        
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

    # Model reconstruction based on quantitative information
    # the acticate/weight quantity must precede this
    model_recon=reconstruction.Reconstruction(model)
    all_quantity_info=model_recon.get_quantity_information()
    quantity_model=model_recon.ReconModel(all_quantity_info)

    # inference of quantity model
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        quantity_model = quantity_model.cuda()

    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        origin_img, label = data
        quantzied_image=torch.round(torch.mul(origin_img,pow(2,7))).clamp(-128.0,127.0)
        if use_gpu:
            img = Variable(quantzied_image, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            img = Variable(quantzied_image, volatile=True)
            label = Variable(label, volatile=True)
        out = quantity_model(img)
        out= torch.div(out,pow(2,2))
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))

if __name__ == "__main__":
    main()


#     from torchvision import transforms
#     from torchvision import datasets
#     import lenet.lenet as lenet
#     from  torch.autograd import Variable
#     from torch.utils.data import DataLoader


#     batch_size = 1
#     test_dataset = datasets.MNIST(
#         root='/private/liusongwei/dataset', train=False, transform=transforms.ToTensor())
#     print(test_dataset)

#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     model = lenet.Cnn(1, 10)
#     model.load_state_dict(torch.load('./lenet/cnn.pth'))

#     use_gpu = torch.cuda.is_available()
#     if use_gpu:
#         model = model.cuda()


#     criterion = nn.CrossEntropyLoss()
#     model=model.eval()
#     eval_loss = 0
#     eval_acc = 0
#     i=0
#     for data in test_loader:
#         i+=1
#         if i>1:
#             break
#         img, label = data
#         if use_gpu:
#             img = Variable(img, volatile=True).cuda()
#             label = Variable(label, volatile=True).cuda()
#         else:
#             img = Variable(img, volatile=True)
#             label = Variable(label, volatile=True)
#         out = model(img)
#         loss = criterion(out, label)
#         eval_loss += loss.item() * label.size(0)
#         _, pred = torch.max(out, 1)
#         num_correct = (pred == label).sum()
#         eval_acc += num_correct.item()
#     print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
#         test_dataset)), eval_acc / (len(test_dataset))))

#     print(model)
#     lenet_model_infor=Get_quantized_information(model,None)
#     print(lenet_model_infor)


    # new_letnetmodel=Reconstruction(model,None,lenet_model_infor).eval()
    # print(new_letnetmodel)
    #
    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     new_letnetmodel = new_letnetmodel.cuda()
    #
    #
    # eval_loss = 0
    # eval_acc = 0
    #
    # i=0
    # for data in test_loader:
    #     # i+=1
    #     # if i>1:
    #     #     break
    #     origin_img, label = data
    #     quantzied_image=torch.round(torch.mul(origin_img,pow(2,7))).clamp(-128.0,127.0)
    #     if use_gpu:
    #         img = Variable(quantzied_image, volatile=True).cuda()
    #         label = Variable(label, volatile=True).cuda()
    #     else:
    #         img = Variable(quantzied_image, volatile=True)
    #         label = Variable(label, volatile=True)
    #     out = new_letnetmodel(img)
    #     out= torch.div(out,pow(2,2))
    #     loss = criterion(out, label)
    #     eval_loss += loss.item() * label.size(0)
    #     _, pred = torch.max(out, 1)
    #     num_correct = (pred == label).sum()
    #     eval_acc += num_correct.item()
    # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    #     test_dataset)), eval_acc / (len(test_dataset))))  

    # new_letnetmodel_test=ReconstructionTest(model,lenet_model_infor).eval()
    
    # print(new_letnetmodel_test)
    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     new_letnetmodel = new_letnetmodel_test.cuda()

    # eval_loss = 0
    # eval_acc = 0

    # i=0
    # for data in test_loader:
    #     # i+=1
    #     # if i>1:
    #     #     break
    #     origin_img, label = data
    #     quantzied_image=torch.div(torch.round(torch.mul(origin_img,pow(2,7))).clamp(-128.0,127.0),pow(2,7))
    #     if use_gpu:
    #         img = Variable(quantzied_image, volatile=True).cuda()
    #         label = Variable(label, volatile=True).cuda()
    #     else:
    #         img = Variable(quantzied_image, volatile=True)
    #         label = Variable(label, volatile=True)
    #     out = new_letnetmodel_test(img)
    #     loss = criterion(out, label)
    #     eval_loss += loss.item() * label.size(0)
    #     _, pred = torch.max(out, 1)
    #     num_correct = (pred == label).sum()
    #     eval_acc += num_correct.item()
    # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    #     test_dataset)), eval_acc / (len(test_dataset))))



