# A Model-driven Deep Neural Network for Single Image Rain Removal
# https://arxiv.org/abs/2005.01333
#http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf

from model import common
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as  F
import os
from torch.autograd import Variable
import torch.distributions.laplace
import scipy.io as io
import numpy as np

def make_model(args, parent=False):
    return Mainnet(args)

# rain kernel  C initialized by the Matlab code "init_rain_kernel.m"
kernel = io.loadmat('./init_kernel.mat')['C9'] # 3*32*9*9
kernel = torch.FloatTensor(kernel)

# filtering on rainy image for initializing B^(0) and Z^(0), refer to supplementary material(SM)
w_x = (torch.FloatTensor([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])/9)
w_x_conv = w_x.unsqueeze(dim=0).unsqueeze(dim=0)  #增加维度

class Mainnet(nn.Module):
    def __init__(self, args):
        super(Mainnet,self).__init__()
<<<<<<< HEAD
        self.S = args.stage                   #阶段号S  #Stage number S includes the initialization process
        self.iter = self.S-1                       #not include the initialization process
=======
        self.S = args.stage                                      #Stage number S includes the initialization process
        self.iter = self.S-1  
        self.kpn = KPN()
                                   #not include the initialization process
>>>>>>> add core, input O,deal with B,rain100L
        self.num_M = args.num_M
        self.num_Z = args.num_Z

        # Stepsize
        self.etaM = torch.Tensor([1])                                   # initialization
        self.etaX = torch.Tensor([5])                                   # initialization
        self.eta1 = nn.Parameter(self.etaM, requires_grad=True)         # usd in initialization process
        self.eta2 = nn.Parameter(self.etaX, requires_grad=True)         # usd in initialization process
        self.eta11 = self.make_eta(self.iter, self.etaM)                # usd in iterative process
        self.eta12 = self.make_eta(self.iter, self.etaX)                # usd in iterative process

        # Rain kernel
        self.weight0 = nn.Parameter(data=kernel, requires_grad = True)  # used in initialization process
        self.conv = self.make_weight(self.iter, kernel)                 # rain kernel is inter-stage sharing. The true net parameter number is (#self.conv /self.iter)

        # filter for initializing B and Z
        self.w_z_f0 = w_x_conv.expand(self.num_Z, 3, -1, -1)
        self.w_z_f = nn.Parameter(self.w_z_f0, requires_grad=True)

        # proxNet in initialization process
        self.xnet = Xnet(self.num_Z+3)                                  # 3 means R,G,B channels for color image
        self.mnet = Mnet(self.num_M)
        self.xnet1 = Xnet(self.num_Z+3)
        self.mnet1 = Mnet(self.num_M)
        # proxNet in iterative process
        self.x_stage = self.make_xnet(self.S, self.num_Z + 3)
        self.m_stage = self.make_mnet(self.S, self.num_M)
        # fine-tune at the last
        self.fmnet = Mnet1(self.num_M)
        self.fxnet = Xnet1(self.num_Z + 3)

        # for sparse rain layer
        self.f = nn.ReLU(inplace=True)
        self.taumm = torch.Tensor([1])
        self.tau = nn.Parameter(self.taumm, requires_grad=True)
      





    def make_xnet(self, iters, channel):
        layers = []
        for i in range(iters):
            layers.append(Xnet(channel))
        return nn.Sequential(*layers)
    def make_mnet(self, iters, num_M):
        layers = []
        for i in range(iters):
            layers.append(Mnet(num_M))
        return nn.Sequential(*layers)
    def make_eta(self, iters,const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters,-1)
        eta = nn.Parameter(data=const_f, requires_grad = True)
        return eta
    def make_weight(self, iters,const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters,-1,-1,-1,-1)
        weight = nn.Parameter(data=const_f, requires_grad = True)
        return weight
            
    def forward(self, input):
        # save mid-updating results
         ListB = []
         ListCM = []
         if self.training is False: 
             input = input[:,:,:-1,:-1]
        # initialize B0 and Z0 (M0 =0)
         z0 = F.conv2d(input, self.w_z_f, stride =1, padding = 1)              # dual variable z with the channels self.num_Z
         input_ini = torch.cat((input, z0), dim=1)
         out_dual = self.xnet(input_ini)
         B0 = out_dual[:,:3,:,:]

         Z = out_dual[:,3:,:,:]

         # 1st iteration: Updating B0-->M1
         ES = input - B0
         ECM = self.f(ES-self.tau)                                            #for sparse rain layer
         GM = F.conv_transpose2d(ECM, self.weight0/10, stride=1, padding=4)   # /10 for controlling the updating speed
         M = self.m_stage[0](GM)

         CM = F.conv2d(M, self.conv[1,:,:,:,:]/10, stride =1, padding = 4)    # self.conv[1,:,:,:,:]：rain kernel is inter-stage sharing
       
         # 1st iteration: Updating M1-->B1
         EB = input - CM
         EX = B0-EB
         GX = EX
         x_dual = B0-self.eta2/10*GX
         input_dual = torch.cat((x_dual, Z), dim=1)
         out_dual = self.x_stage[0](input_dual)
         B = out_dual[:,:3,:,:]

         Z = out_dual[:,3:,:,:]
         ListB.append(B)
         ListCM.append(CM)

         for i in range(self.iter-1):
             # M-net
             ES = input - B
             ECM = CM- ES
             GM = F.conv_transpose2d(ECM,  self.conv[1,:,:,:,:]/10, stride =1, padding = 4)
             input_new = M - self.eta11[i,:]/10*GM
             M = self.m_stage[i+1](input_new)
             # B-net
             CM = F.conv2d(M,  self.conv[1,:,:,:,:]/10, stride =1, padding = 4)
             ListCM.append(CM)
             EB = input - CM
             EX = B - EB
             GX = EX
             x_dual = B - self.eta12[i,:]/10*GX
             input_dual = torch.cat((x_dual,Z), dim=1) 
             out_dual  = self.x_stage[i+1](input_dual)
             B = out_dual[:,:3,:,:]
             Z = out_dual[:,3:,:,:]
             ListB.append(B)

         ES = input - B
         ECM = CM- ES
         GM = F.conv_transpose2d(ECM,  self.conv[1,:,:,:,:]/10, stride =1, padding = 4)
         input_new = M - self.eta11[i,:]/10*GM
         M = self.fmnet(input_new)
             # B-net
         CM = F.conv2d(M,  self.conv[1,:,:,:,:]/10, stride =1, padding = 4)
         ListCM.append(CM)
         EB = input - CM
         EX = B - EB
         GX = EX
         x_dual = B - self.eta12[i,:]/10*GX
         input_dual = torch.cat((x_dual,Z), dim=1) 
         out_dual  = self.x_stage[self.iter](input_dual)
         B = out_dual[:,:3,:,:]
         Z = out_dual[:,3:,:,:]
         ListB.append(B)

         out_dual = self.fxnet(out_dual)                # fine-tune
         B = out_dual[:,:3,:,:]
         B = self.kpn(input,B,white_level=1.0)
         ListB.append(B)
         return B0, ListB, ListCM

# proxNet_M
class Mnet(nn.Module):
    def __init__(self, channels):
        super(Mnet, self).__init__()
        self.channels = channels
        self.tau0 = torch.Tensor([0.5])
        self.taum= self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, self.channels,-1,-1)
        self.tau = nn.Parameter(self.taum, requires_grad=True)                      # for sparse rain map
        self.f = nn.ReLU(inplace=True)
        self.resm1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation =1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                   )
        self.resm2 = nn.Sequential(nn.Conv2d(self.channels, self.channels,kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resm3 = nn.Sequential(nn.Conv2d(self.channels, self.channels,kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resm4 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                   )
    def forward(self, input):
        m1  = F.relu(input + self.resm1(input))
        m2  = F.relu(m1+ self.resm2(m1))
        m3  = F.relu(m2+ self.resm3(m2))
        m4  = F.relu(m3+ self.resm4(m3))
        m_rev =self.f(m4-self.tau)                                     # for sparse rain map
        return m_rev



class Xnet(nn.Module):
    def __init__(self, channels):
        super(Xnet, self).__init__()
        self.channels = channels
        self.resx1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resx2 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resx3 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                 nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels,  kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resx4 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
    def forward(self, input):
        x1  = F.relu(input + self.resx1(input))
        x2  = F.relu(x1 + self.resx2(x1))
        x3  = F.relu(x2 + self.resx3(x2))
        x4  = F.relu(x3 + self.resx4(x3))
        return x4


<<<<<<< HEAD
# proxNet_M
class Mnet1(nn.Module):
    def __init__(self, channels,color=True, burst_length=1, blind_est=True, kernel_size2=[3], sep_conv=False,core_bias=False):
        super(Mnet1, self).__init__()



        self.burst_length = burst_length
        self.core_bias = core_bias

        self.channels = channels
        self.tau0 = torch.Tensor([0.5])
        self.taum= self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, self.channels,-1,-1)
        self.tau = nn.Parameter(self.taum, requires_grad=True) 
        # 首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter
        #   所以在参数优化的时候可以进行优化的)，所以经过类型转换这个self.v变成了模型的一部分，
        # 成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。



                             # for sparse rain map
        self.f = nn.ReLU(inplace=True)
        self.resm1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation =1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                   )

        self.resm2 = nn.Sequential(nn.Conv2d(self.channels, self.channels,kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )

        self.resm3 = nn.Sequential(nn.Conv2d(self.channels, self.channels,kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )

        self.resm4 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        
        self.outc = nn.Conv2d(self.channels,self.channels*9, kernel_size=1, stride = 1,padding = 0)
        self.kernel_pred = KernelConv(kernel_size2, sep_conv, self.core_bias)
        self.conv_final = nn.Conv2d(self.channels*4,self.channels, kernel_size=3, stride=1, padding=1)


              
    def forward(self, input,white_level=1.0):
        m1  = F.relu(input + self.resm1(input))
        m2  = F.relu(m1+ self.resm2(m1))
        m3  = F.relu(m2+ self.resm3(m2))
        m4  = F.relu(m3+ self.resm4(m3))
        m5 =self.f(m1-self.tau)                                     # for sparse rain map
        
        core = self.outc(m5)
        
        pred1 = self.kernel_pred(input, core, white_level, rate=1)
        pred2 = self.kernel_pred(input, core, white_level, rate=2)
        pred3 = self.kernel_pred(input, core, white_level, rate=3)
        pred4 = self.kernel_pred(input, core, white_level, rate=4)

        pred_cat = torch.cat([torch.cat([torch.cat([pred1, pred2], dim=1), pred3], dim=1), pred4], dim=1)
        m_rev = self.conv_final(pred_cat)
        
        #pred = self.kernel_pred(data, core, white_level, rate=1)
        
        return m_rev

# proxNet_B
class Xnet1(nn.Module):
    def __init__(self, channels,color=True, burst_length=1, blind_est=True, kernel_size2=[3], sep_conv=False,core_bias=False):
        super(Xnet1, self).__init__()
        self.burst_length = burst_length
              
        self.core_bias = core_bias
        self.channels = channels
        self.resx1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resx2 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resx3 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                 nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels,  kernel_size=3, stride = 1, padding= 1, dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.resx4 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 1,  dilation = 1),
                                  nn.BatchNorm2d(self.channels),
                                  )
        self.outc = nn.Conv2d(self.channels,self.channels*9, kernel_size=1, stride = 1,padding = 0)
        self.kernel_pred = KernelConv(kernel_size2, sep_conv, self.core_bias)
        self.conv_final = nn.Conv2d(self.channels*4,self.channels, kernel_size=3, stride=1, padding=1)

          
    def forward(self, input,white_level=1.0):
        x1  = F.relu(input + self.resx1(input))
        x2  = F.relu(x1 + self.resx2(x1))
        x3  = F.relu(x2 + self.resx3(x2))
        x4  = F.relu(x3 + self.resx4(x3))
        
        core = self.outc(x4)
        
        pred1 = self.kernel_pred(input, core, white_level, rate=1)
        pred2 = self.kernel_pred(input, core, white_level, rate=2)
        pred3 = self.kernel_pred(input, core, white_level, rate=3)
        pred4 = self.kernel_pred(input, core, white_level, rate=4)

        pred_cat = torch.cat([torch.cat([torch.cat([pred1, pred2], dim=1), pred3], dim=1), pred4], dim=1)
        pred = self.conv_final(pred_cat)
        
=======




class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2*out_ch, out_ch//g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch//g, out_ch, 1, 1, 0),
                nn.Sigmoid()
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid()
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            # fm_pool = F.adaptive_avg_pool2d(fm, (1, 1)) + F.adaptive_max_pool2d(fm, (1, 1))
            fm_pool = torch.cat([F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1)
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat([torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1)
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm

class KPN(nn.Module):
    def __init__(self, color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
                 #Bilinear Interpolation
#插值根据于待求点P最近4个点的像素值，计算出P点的像素值。
    #这个卷积名字起得花里胡哨的，其实总结起来就是输入通道每个通道一个卷积得到和输入通道数相同的特征图，然后再使用若干个1*1的卷积聚合每个特征图的值得到输出特征图。
#假设我们输入通道是16，输出特征图是32，并且使用3*3的卷积提取特征，那么第一步一共需要16*3*3个参数，第二步需要32*16*1*1个参数，一共需要16*3*3+32*16*1*1=656个参数。

        super(KPN, self).__init__()#super().__init__(),就是继承父类的init方法
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length+1)
        out_channel = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512+512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256+512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256+128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)
        
        self.conv_final = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
       # if self.training is False:
        #    print('conv1:',conv1.size())
         #   print('conv2:',conv2.size())
          #  print('conv3:',conv3.size())
           # print('conv4:',conv4.size())
            #print('conv5:',conv5.size())
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))#dim = 1 ,通道相加
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
        #print(conv7.size())
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*N
        core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode))
       
        pred1 = self.kernel_pred(data, core, white_level, rate=1)#white_level=1.0
        pred2 = self.kernel_pred(data, core, white_level, rate=2)
        pred3 = self.kernel_pred(data, core, white_level, rate=3)
        pred4 = self.kernel_pred(data, core, white_level, rate=4)

        pred_cat = torch.cat([torch.cat([torch.cat([pred1, pred2], dim=1), pred3], dim=1), pred4], dim=1)
        
        pred = self.conv_final(pred_cat)
>>>>>>> add core, input O,deal with B,rain100L
        #pred = self.kernel_pred(data, core, white_level, rate=1)
        
        return pred

<<<<<<< HEAD

        





=======
>>>>>>> add core, input O,deal with B,rain100L
class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
<<<<<<< HEAD
    def __init__(self, kernel_size2=[3], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size2 = sorted(kernel_size2)
=======
    def __init__(self, kernel_size=[3], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size) #sorted() 函数对所有可迭代的对象进行排序操作
>>>>>>> add core, input O,deal with B,rain100L
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
<<<<<<< HEAD
        kernel_total = sum(self.kernel_size2)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
=======
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2) # torch.split()作用将tensor分成块结构 
            #kernel_total：需要切分的大小(int or list )
# split_size_or_sections 为切分后的每块大小，不是切分为多少块.当split_size_or_sections为int时，tenor结构和split_size_or_sections，正好匹配，那么ouput就是大小相同的块结构。
#如果按照split_size_or_sections结构，tensor不够了，那么就把剩下的那部分做一个块处理。
# 当split_size_or_sections 为list时，那么tensor结构会一共切分成len(list)这么多的小块，
#每个小块中的大小按照list中的大小决定，其中list中的数字总和应等于该维度的大小，否则会报错
#（注意这里与split_size_or_sections为int时的情况不同）。
#https://blog.csdn.net/qq_42518956/article/details/103882579

>>>>>>> add core, input O,deal with B,rain100L
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
<<<<<<< HEAD
        for K in self.kernel_size2:
=======
        for K in self.kernel_size:
>>>>>>> add core, input O,deal with B,rain100L
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
<<<<<<< HEAD
=======
            #torch.einsum ('ijklno,ijlmno->ijkmno')
            # Viewed A: torch.Size([16, 8, 5, 1, 128, 128])
            # Viewed B: torch.Size([16, 8, 1, 5, 128, 128])
            # C: torch.Size([16, 8, 5, 5, 128, 128])
>>>>>>> add core, input O,deal with B,rain100L
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
<<<<<<< HEAD
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size2[0]] = core[:, :, 0:self.kernel_size2[0]**2, ...]
=======
       # if self.training is False:
        #    core = core.view(batch_size, N, -1, color, height-1, width-1) 
         #   print(core.size())
       # else:
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        core_out[self.kernel_size[0]] = F.softmax(core_out[self.kernel_size[0]],dim=2)
>>>>>>> add core, input O,deal with B,rain100L
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0, rate=1):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        if len(frames.size()) == 5:
<<<<<<< HEAD
            batch_size, N, color, height, width = frames.size()
=======
           
                  
            batch_size, N, color, height, width = frames.size()
          #  if self.training is False:
            #     frames = frames[:,:,:,:height-1,:width-1]
              #   pritn(frames.size())
>>>>>>> add core, input O,deal with B,rain100L
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
<<<<<<< HEAD
=======
           # if self.training is False:
            #    frames = frames[:,:,:,:height-1,:width-1]
             #   print("frames:",frames.size())
>>>>>>> add core, input O,deal with B,rain100L
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        img_stack = []
        pred_img = []
<<<<<<< HEAD
        kernel = self.kernel_size2[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:
                padding_num = (K//2) * rate
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])
                for i in range(0, K):
                    for j in range(0, K):
                        img_stack.append(frame_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
                img_stack = torch.stack(img_stack, dim=2)
=======
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)
                                            #组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
            if not img_stack:
                padding_num = (K//2) * rate
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])#pad 扩充
                for i in range(0, K):
                    for j in range(0, K):
                       # if self.training is False:
                          #  img_stack.append(frame_pad[..., i*rate:i*rate + height-1, j*rate:j*rate + width-1])
                       # else:
                        img_stack.append(frame_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
                img_stack = torch.stack(img_stack, dim=2)
                #浅显说法：把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠。
>>>>>>> add core, input O,deal with B,rain100L
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            # print('img_stack:', img_stack.size())
            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))
        pred_img = torch.stack(pred_img, dim=0)
        # print('pred_stack:', pred_img.size())
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        #print("pred_img_i", pred_img_i.size())
        # N = 1
        pred_img_i = pred_img_i.squeeze(2)
        #print("pred_img_i", pred_img_i.size())
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i += bias
        # print('white_level', white_level.size())
        pred_img_i = pred_img_i / white_level
        #pred_img = torch.mean(pred_img_i, dim=1, keepdim=True)
        # print('pred_img:', pred_img.size())
        # print('pred_img_i:', pred_img_i.size())
        return pred_img_i
<<<<<<< HEAD
=======

>>>>>>> add core, input O,deal with B,rain100L
