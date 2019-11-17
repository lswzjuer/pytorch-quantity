import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt


QUANTIZE_BIT=8

# right shift
class RightShift(nn.Module):
    def __init__(self,bits,rs):
        super(RightShift, self).__init__()
        self.rs=rs
        self.Bit_width=bits

    def forward(self, x):
        sh=x.size()
        input_flatten=x.view(-1)

        assert self.Bit_width==8 or self.Bit_width==16, "Not support bit width."

        if self.Bit_width==8:
            max_val=float(127)
            min_val=float(-128)
        else:
            max_val=float(32767)
            min_val=float(-32768)
        # right shift
        input_shift=torch.div(input_flatten,pow(2,self.rs))
        # round
        gtmap=torch.gt(input_shift,0).type(torch.float32)
        lemap=torch.le(input_shift,0).type(torch.float32)
        add_gt=torch.mul(gtmap,0.5)
        add_le=torch.mul(lemap,-0.5)
        add_nums=torch.add(add_gt,add_le)
        input_round=torch.add(input_shift,add_nums).type(torch.int32)
        #input_round=torch.round(input_shift)

        # clamp->[min,max]
        output_flatten=torch.clamp(input_round, int(min_val), int(max_val)).type(torch.float32)

        # reshape
        return output_flatten.reshape(sh)


# Quantization input vector
class Quantity(nn.Module):
    def __init__(self,ib):
        super(Quantity, self).__init__()
        self.ib=ib
    def forward(self,x):
        if QUANTIZE_BIT==8:
            output=torch.round(torch.mul(x, pow(2, self.ib))).clamp(-128.0, 127.0)
        else:
            output=torch.round(torch.mul(x, pow(2, self.ib))).clamp(-32768.0, 32767.0)
        #output=torch.mul(x, pow(2, self.ib))
        return output

# Inverse quantization output vector
class DeQuantity(nn.Module):
    """docstring for DeQuantity"""
    def __init__(self, ob):
        super(DeQuantity, self).__init__()
        self.ob = ob
    def forward(self,x):     
        output=torch.div(x,pow(2,self.ob))
        return output       

# Saturated truncation
class Sp(nn.Module):
    def __init__(self,bits):
        super(Sp, self).__init__()
        self.bitwidth=bits

    def forward(self, x):
        sh = x.shape
        input_flatten = x.view(-1)

        assert self.bitwidth==8 or self.bitwidth==16, "Not support bit width."

        if self.bitwidth == 8:
            max_val = float(127.0)
            min_val = float(-128.0)
        else:
            max_val = float(32767.0)
            min_val = float(-32768.0)

        # clmap
        output_flatten=torch.clamp(input_flatten,min=min_val,max=max_val)
        return output_flatten.reshape(sh)


# Convolution result plus bias
class BiasAdd(nn.Module):
    def __init__(self):
        super(BiasAdd, self).__init__()
        
    def forward(self, x, y):
        output = torch.add(x, y)
        return output

# Convolution node of a quantization operation
class NewConv2d(nn.Module):
    def __init__(self,conv_module,quantize_infor):
        super(NewConv2d, self).__init__()
        self.weight_bit=quantize_infor['weight_bit']
        self.bias_bit=quantize_infor['bias_bit']
        self.input_bit=quantize_infor['input_bit']
        self.output_bit=quantize_infor['output_bit']
        self.rs_bit=self.weight_bit+self.input_bit-self.output_bit

        # model structure
        self.Quan=Quantity(self.input_bit)
        self.Conv=conv_module
        self.RightShift=RightShift(QUANTIZE_BIT,self.rs_bit)
        self.BiasAdd=BiasAdd()
        self.Sp=Sp(QUANTIZE_BIT)
        self.DeQuan=DeQuantity(self.output_bit)

        # quantize weights and biases
        self.quantity()

    def forward(self, input):
        conv_output=self.Quan(input)
        conv_output=self.Conv(conv_output)
        conv_output=self.RightShift(conv_output)
        self.quantized_bias_expand=self.quantized_bias.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand_as(conv_output)

        conv_output=self.BiasAdd(conv_output,self.quantized_bias_expand)
        conv_output=self.Sp(conv_output)
        conv_output=self.DeQuan(conv_output)
        return conv_output

    def quantity(self):
        self.weight=self.Conv.weight
        self.bias=self.Conv.bias
        # 经过BN层融合之后bias不会为0，但是在这里考虑兼容这种情况
        assert type(self.weight).__name__ != "NoneType", "The conv weight can`t be None"
        weight_data=self.weight.data
        if type(self.bias).__name__ == "NoneType":
            bias_data =torch.zeros(size=[self.Conv.out_channels])
        else:
            bias_data=self.bias.data

        # quantize
        quantized_weight=torch.round(torch.mul(weight_data,pow(2,self.weight_bit)))
        quantized_bias=torch.round(torch.mul(bias_data,pow(2,self.bias_bit)))

        if QUANTIZE_BIT==8:
            quantized_weight_clmap=quantized_weight.clamp(-128,127)
            quantized_bias_clmap=quantized_bias.clamp(-128,127)
        else:
            quantized_weight_clmap=quantized_weight.clamp(-128,127)
            quantized_bias_clmap=quantized_bias.clamp(-32768.0,32767.0)

        # modify the conv layer weight and bias
        # ( this bias should be zero, because we split the matmul and add operation)
        self.Conv.weight=nn.Parameter(quantized_weight_clmap)
        self.Conv.bias=nn.Parameter(torch.zeros(size=[self.Conv.out_channels]))

        # original bias parameter, we use it in BiasAdd layer
        self.quantized_bias=quantized_bias_clmap

# The sum operation of quantized operations
class NewAdd(nn.Module):
    def __init__(self):
        super(NewAdd, self).__init__()
        self.Sp=Sp(QUANTIZE_BIT)

    def forward(self, x,y ):
        out_put=torch.add(x,y)
        out_put=self.Sp(out_put)
        return out_put

# Full connectivity layer for quantization operations
class NewLinear(nn.Module):
    def __init__(self,linear_module,quantize_infor):
        super(NewLinear, self).__init__()
        self.weight_bit=quantize_infor['weight_bit']
        self.bias_bit=quantize_infor['bias_bit']
        self.input_bit=quantize_infor['input_bit']
        self.output_bit=quantize_infor['output_bit']
        self.rs_bit=self.weight_bit+self.input_bit-self.output_bit

        # model structure
        self.Quan=Quantity(self.input_bit)
        self.Linear=linear_module
        self.RightShift=RightShift(QUANTIZE_BIT,self.rs_bit)
        self.BiasAdd=BiasAdd()
        self.Sp=Sp(QUANTIZE_BIT)
        self.DeQuan=DeQuantity(self.output_bit)

        # quantize weights and biases
        self.quantity()

    def forward(self, input):
        linear_output=self.Quan(input)
        linear_output=self.Linear(linear_output)
        linear_output=self.RightShift(linear_output)
        self.quantized_bias_expand=self.quantized_bias.unsqueeze(0).expand_as(linear_output)
        linear_output=self.BiasAdd(linear_output,self.quantized_bias_expand)
        linear_output=self.Sp(linear_output)
        linear_output=self.DeQuan(linear_output)
        return linear_output


    def quantity(self):
        self.weight=self.Linear.weight
        self.bias=self.Linear.bias
        # 经过BN层融合之后bias不会为0，但是在这里考虑兼容这种情况
        assert type(self.weight).__name__ != "NoneType", "The linear layer weight can`t be None"
        weight_data=self.weight.data
        if type(self.bias).__name__ == "NoneType":
            bias_data =torch.zeros(size=[self.Linear.out_features])
        else:
            bias_data=self.bias.data

        # quantize
        quantized_weight=torch.round(torch.mul(weight_data,pow(2,self.weight_bit)))
        quantized_bias=torch.round(torch.mul(bias_data,pow(2,self.bias_bit)))

        if QUANTIZE_BIT==8:
            quantized_weight_clmap=quantized_weight.clamp(-128,127)
            quantized_bias_clmap=quantized_bias.clamp(-128,127)
        else:
            quantized_weight_clmap=quantized_weight.clamp(-32768,32767)
            quantized_bias_clmap=quantized_bias.clamp(-32768.0,32767.0)

        # modify the conv layer weight and bias
        # ( this bias should be zero, because we split the matmul and add operation)
        self.Linear.weight=nn.Parameter(quantized_weight_clmap)
        self.Linear.bias=nn.Parameter(torch.zeros(size=[self.Linear.out_features]))

        # original bias parameter, we use it in BiasAdd layer
        self.quantized_bias=quantized_bias_clmap


class QuanDequan(nn.Module):
    """docstring for QuanDequan"""
    def __init__(self,Bitwidth,bit):
        super(QuanDequan, self).__init__()
        self.bitwidth=Bitwidth
        self.bit=bit

    def forward(self,quantized_x):
        # quantized result
        quantized_x=torch.round(  torch.mul(  quantized_x, pow(2,self.bit)  )   )
        if self.bitwidth==8:
            quantized_x=quantized_x.clamp(-128,127)
            #quantized_x = quantized_x
        else:
            quantized_x=quantized_x.clamp(-32768.0,32767.0)
            #quantized_x = quantized_x
        #dequantized
        quantized_x=torch.div( quantized_x, pow(2,self.bit) )
        return quantized_x

class TestConv(nn.Module):
    """docstring for TestConv"""
    def __init__(self, name, module, quantize_infor,new_model_path):
        super(TestConv, self).__init__()
        self.name = name

        # create the quantity and dequantity results folder
        self.path=os.path.join(os.path.dirname(new_model_path),"quantity_results")
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.weight_bit=quantize_infor['weight_bit']
        self.bias_bit=quantize_infor['bias_bit']
        self.input_bit=quantize_infor['input_bit']
        self.output_bit=quantize_infor['output_bit']
        self.weight_qdp=QuanDequan(QUANTIZE_BIT,self.weight_bit)
        self.bias_qdp=QuanDequan(QUANTIZE_BIT,self.bias_bit)
        self.output_qdp=QuanDequan(QUANTIZE_BIT,self.output_bit)
        self.Conv=module
        self.feature_extract()

    def forward(self,x):

        # 通过这种方法来衡量 输入 权重 输出 三种量化哪个对精度的影响更大
        output=self.Conv(x)
        output_qdp=self.output_qdp(output)

        # #Save the output of quantization and inverse quantization
        # output_numpy="  ".join([str(x) for x in output.cpu().detach().numpy().flatten()])
        # output_qdp_numpy="  ".join([str(x) for x in output_qdp.cpu().detach().numpy().flatten()])
        # with open(os.path.join(self.path,self.name.replace('.','_')+"_output.txt"),'a') as file:
        #     file.write(output_numpy+'\n')
        #     file.write(output_qdp_numpy+'\n')
        return output_qdp

    def feature_extract(self):
        self.weight=self.Conv.weight
        self.bias=self.Conv.bias
        # 经过BN层融合之后bias不会为0，但是在这里考虑兼容这种情况
        assert type(self.weight).__name__ != "NoneType", "The Conv layer weight can`t be None"
        weight_data=self.weight.data
        if type(self.bias).__name__ == "NoneType":
            bias_data =torch.zeros(size=[self.Linear.out_features])
        else:
            bias_data=self.bias.data

        weight_data_qdp=self.weight_qdp(weight_data)
        bias_data_qdp=self.bias_qdp(bias_data)

        self.Conv.weight=nn.Parameter(weight_data_qdp)
        self.Conv.bias=nn.Parameter(bias_data_qdp)


        # visualization code
        # save origin weight and bias
        weight_data_numpy="  ".join([str(x) for x in weight_data.cpu().numpy().flatten()])
        bias_data_numpy="  ".join([str(x) for x in bias_data.cpu().numpy().flatten()])

        # quantize-dequantize weight and bias numpy
        weight_data_qdp_numpy="  ".join([str(x) for x in weight_data_qdp.cpu().numpy().flatten()])
        bias_data_qdp_numpy="  ".join([str(x) for x in bias_data_qdp.cpu().numpy().flatten()])

        with open(os.path.join(self.path, self.name.replace('.','_') + "_weight.txt"), 'a') as file:
            file.write(weight_data_numpy+'\n\n\n\n')
            file.write(weight_data_qdp_numpy+'\n\n\n\n')
        with open(os.path.join(self.path,self.name.replace('.','_')+"_bias.txt"),'a') as file:
            file.write(bias_data_numpy+'\n\n\n\n')
            file.write(bias_data_qdp_numpy+'\n\n\n\n')

        # 画出量化和原始权重的直方图
        weight_path=os.path.join(self.path, self.name.replace('.','_') + "_weight_o.png")
        bias_path=os.path.join(self.path, self.name.replace('.','_') + "_bias_o.png")
        self.plot_hist(weight_data.cpu().numpy(),2048,weight_path,title='weight')
        self.plot_hist(bias_data.cpu().numpy(),2048,bias_path,title='bias')

        weight_path_q=os.path.join(self.path, self.name.replace('.','_') + "_weight_q.png")
        bias_path_q=os.path.join(self.path, self.name.replace('.','_') + "_bias_q.png")
        self.plot_hist(weight_data_qdp.cpu().numpy(),2048,weight_path_q,title='weight')
        self.plot_hist(bias_data_qdp.cpu().numpy(),2048,bias_path_q,title='bias')


    # Drawing histogram
    def plot_hist(self,ndarray, bins, save_path, title):

        data=ndarray.flatten()
        # set the plt
        fig=plt.figure()
        plt.grid()
        plt.title(title)
        plt.xlabel('bins')
        plt.ylabel('counter/frequency')
        plt.hist(data, bins,density=True, histtype='bar', facecolor='blue')
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        else:
            raise NotImplementedError("the path is not exists")
        plt.close(fig)


class TestLinear(nn.Module):
    def __init__(self, name, module, quantize_infor,new_model_path):
        super(TestLinear, self).__init__()
        self.name = name

        # create the quantity and dequantity results folder
        self.path=os.path.join(os.path.dirname(new_model_path),"quantity_results")
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.weight_bit=quantize_infor['weight_bit']
        self.bias_bit=quantize_infor['bias_bit']
        self.input_bit=quantize_infor['input_bit']
        self.output_bit=quantize_infor['output_bit']
        self.weight_qdp=QuanDequan(QUANTIZE_BIT,self.weight_bit)
        self.bias_qdp=QuanDequan(QUANTIZE_BIT,self.bias_bit)
        self.output_qdp=QuanDequan(QUANTIZE_BIT,self.output_bit)
        self.linear=module
        self.feature_extract()

    def forward(self,x):
        output=self.linear(x)
        output_qdp=self.output_qdp(output)

        # Save the output of quantization and inverse quantization
        # output_numpy="  ".join([str(x) for x in output.cpu().detach().numpy().flatten()])
        # output_qdp_numpy="  ".join([str(x) for x in output_qdp.cpu().detach().numpy().flatten()])
        # with open(os.path.join(self.path,self.name.replace('.','_')+"_output.txt"),'a') as file:
        #     file.write(output_numpy+'\n')
        #     file.write(output_qdp_numpy+'\n')

        return output_qdp


    def feature_extract(self):
        self.weight=self.linear.weight
        self.bias=self.linear.bias
        # 经过BN层融合之后bias不会为0，但是在这里考虑兼容这种情况
        assert type(self.weight).__name__ != "NoneType", "The Conv layer weight can`t be None"
        weight_data=self.weight.data
        if type(self.bias).__name__ == "NoneType":
            bias_data =torch.zeros(size=[self.Linear.out_features])
        else:
            bias_data=self.bias.data

        weight_data_qdp=self.weight_qdp(weight_data)
        bias_data_qdp=self.bias_qdp(bias_data)

        self.linear.weight=nn.Parameter(weight_data_qdp)
        self.linear.bias=nn.Parameter(bias_data_qdp)


        # visualization code
        # save origin weight and bias
        weight_data_numpy="  ".join([str(x) for x in weight_data.cpu().numpy().flatten()])
        bias_data_numpy="  ".join([str(x) for x in bias_data.cpu().numpy().flatten()])

        # quantize-dequantize weight and bias numpy
        weight_data_qdp_numpy="  ".join([str(x) for x in weight_data_qdp.cpu().numpy().flatten()])
        bias_data_qdp_numpy="  ".join([str(x) for x in bias_data_qdp.cpu().numpy().flatten()])

        with open(os.path.join(self.path, self.name.replace('.','_') + "_weight.txt"), 'a') as file:
            file.write(weight_data_numpy+'\n\n\n\n')
            file.write(weight_data_qdp_numpy+'\n\n\n\n')
        with open(os.path.join(self.path,self.name.replace('.','_')+"_bias.txt"),'a') as file:
            file.write(bias_data_numpy+'\n\n\n\n')
            file.write(bias_data_qdp_numpy+'\n\n\n\n')

        # 画出量化和原始权重的直方图
        weight_path=os.path.join(self.path, self.name.replace('.','_') + "_weight_o.png")
        bias_path=os.path.join(self.path, self.name.replace('.','_') + "_bias_o.png")
        self.plot_hist(weight_data.cpu().numpy(),2048,weight_path,title='weight')
        self.plot_hist(bias_data.cpu().numpy(),2048,bias_path,title='bias')

        weight_path_q=os.path.join(self.path, self.name.replace('.','_') + "_weight_q.png")
        bias_path_q=os.path.join(self.path, self.name.replace('.','_') + "_bias_q.png")
        self.plot_hist(weight_data_qdp.cpu().numpy(),2048,weight_path_q,title='weight')
        self.plot_hist(bias_data_qdp.cpu().numpy(),2048,bias_path_q,title='bias')

    # 画直方图的函数
    def plot_hist(self,ndarray,bins,save_path,title):

        data=ndarray.flatten()
        fig=plt.figure()
        # set the plt
        plt.grid()
        plt.title(title)
        plt.xlabel('bins')
        plt.ylabel('counter/frequency')
        plt.hist(data,bins,density=True,histtype='bar',facecolor='blue')
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        else:
            raise NotImplementedError("the path is not exists")
        plt.close(fig)


if __name__=="__main__":

    test_data=torch.zeros((2,3))
    print(test_data)

    test_data2=test_data.view(-1)
    print(test_data2)

    size=test_data.size()
    size_str=""
    for i in size:
        size_str+=str(i)
    print(size_str)

    for i in test_data2:
        if i >0:
            print("True")
        else:
            print("False")

    for i in range(len(test_data2)):
        test_data2[i]=float(129)
    print(test_data2)
    print(type(test_data2))

    test_data3=test_data.clone()
    print(test_data3)

    test_data4=torch.randint(-500,300,size=(2,3),dtype=torch.float32)
    print(test_data4)
    gtnums=torch.gt(test_data4,0).type(torch.float32)
    print(gtnums)
    print(type(gtnums))
    rev_nums=torch.le(test_data4,0).type(torch.float32)
    print(rev_nums)
    print(type(gtnums))
    add_grtnums=torch.mul(gtnums,0.5)
    add_revnums=torch.mul(rev_nums,-0.5)

    print(add_grtnums)
    print(add_revnums)
    test_add=torch.add(add_grtnums,add_revnums)
    print(test_add)
    test_data6=torch.add(test_data4,test_add)
    print(test_data6)

    int_testdata=test_data6.type(torch.int32)
    print(int_testdata)

    clamp_data=torch.clamp(int_testdata,-127,128).type(torch.float32)
    print(clamp_data)

    # 测试nn.Parameter能不能直接进行运算
    test_bias_data_rand=torch.randn(size=(10,3,12,12))
    conv_layer=nn.Conv2d(3,3,(3,3),1,0,1,1,False)
    out_put_data=conv_layer(test_bias_data_rand)
    print(out_put_data)

    test_weight=conv_layer.weight
    test_bias=conv_layer.bias
    #
    # print(test_weight)
    # print(test_bias)
    #
    #运算
    weight_quantity =torch.round(torch.mul(test_weight,pow(2,1)))
    param_quantity = weight_quantity.clamp(-128,127)
    print(test_weight)
    print(weight_quantity)
    # conv_layer.weight=weight_quantity

    #
    # bias_quantity =torch.round(torch.mul(test_bias,pow(2,1)))
    # bias_quantity_ = weight_quantity.clamp(-128,127)
    # print(bias_quantity)
    # print(bias_quantity_)
    # print(weight_quantity.dtype)

    conv_layer.bias=nn.Parameter(torch.ones(size=[conv_layer.out_channels]))
    new_bias=conv_layer.bias
    # print(new_bias)
    # print(new_bias.dtype)

    print(conv_layer(test_bias_data_rand))

    test_data=torch.randn(size=(3,500))
    linear_layer=nn.Linear(500,100,bias=True)
    weight=linear_layer.weight
    bias=linear_layer.bias
    print(weight)
    print(bias)
    print(linear_layer.in_features)
    print(linear_layer.out_features)
    linear_layer.bias=bias
    print(linear_layer)
    print(linear_layer.bias)







