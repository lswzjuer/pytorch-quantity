# 项目简介
pytorchremodel 是针对pytorch训练出的模型提供量化，和仿真的工具包，目的是针对fabu芯片提供量化工具。
主要功能有：
1. 量化。将现有模型进行量化，输出激活层小数位（feat.table文件），权重小数位（weight.table文件）。量化后权重（json文件）
2. 仿真推理。模拟芯片的模型推理功能，利用小数位等信息对模型进行推理，查看其量化后精度

# 项目路径

pytorchremodel
I── quantity  
I   I── common  
I   I── model  
I   I── test  
I   L── tools  
L── README.md  

model: 存储训练好的模型（结构文件: xx.py, 权重文件 xx.pth）  
common: 量化所需的运算与量化层定义  
tool: 量化工具函数和量化配置  
test：用户路径，量化与测试时用户需要写的代码与配置  

# 使用说明

## 量化
1. 将训练好的模型（a.结构文件: xx.py, b.权重文件 xx.pth ）,存放如model目录中
2. 在test中user_configs.yml设置：
（1）数据路径，DATA_PATH
（2）模型路径，MODEL_NET_PATH(.py),MODEL_PATH(.pth)
（2）输入shape ,INPUT_SHAPE
（3）数据模式：0：图片读取，并设置MEAN,RESIZE,SCALE; 1：dataloard  2:npy文件
（4）gpu or cpu :DEVICE
3. 编写量化程序，参考quantity_resnet.py，注意需要自己生成对象传入接口

## 仿真推理

1. 将训练好的模型（a.结构文件: xx.py, b.权重文件 xx.pth ）,存放如model目录中
2. 在test中user_configs.yml设置：
（1）数据路径，DATA_PATH
（2）模型路径，MODEL_NET_PATH(.py),MODEL_PATH(.pth)

3. 编写仿真推理程序，参考resnet_reconstruction.py，注意仿真接口返回量化模型，具体数据预处理与精度计算需用户自己编写。

## NOTE: 

1. 本工具支持量化的模型必须基于common.quantity中fabu_layer.py中提供的module编写
   并且整个模型中所有的运算都必须是基于nn.Module类，不能结构中引入无注册的运算（比如
   使用torch.nn.functional的函数）。

2. 量化->推理，两个过程均基于无BN层的模型，所以量化之前请先使用提供的函数融合BN层。

3. 目前量化节点仅支持 Conv2d, Linear, Eltwise, Concat, 对于其他带可训练参数的节点的
    支持可以参考已有模块，在common.quantity中new_new_quantity_op.py中添加。

4. 基于量化得到的小数位信息，inference的时候提供了两种实质等同的方案：ReconModel&ReconTest
   前者模拟芯片运行情况，卷积等运算是量化后的整数参与。 后者对参数进行量化后立刻反量化，卷积等
   运算是截取了精度的浮点数参与，后者提供了可视化代码，可以生成各层权重参数的数值和直方图信息，查看量化效果。

5. inference过程中会保存量化完成的模型，该模型加载运行的时候注意要import module定义文件。
