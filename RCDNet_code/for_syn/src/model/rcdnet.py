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
        self.S = args.stage                   #阶段号S  #Stage number S includes the initialization process
        self.iter = self.S-1                       #not include the initialization process
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
        m5 =self.f(m4-self.tau)                                     # for sparse rain map
        
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
        
        #pred = self.kernel_pred(data, core, white_level, rate=1)
        
        return pred


        





class KernelConv(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self, kernel_size2=[3], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size2 = sorted(kernel_size2)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        kernel_total = sum(self.kernel_size2)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size2:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum('ijklno,ijlmno->ijkmno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size2[0]] = core[:, :, 0:self.kernel_size2[0]**2, ...]
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
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        img_stack = []
        pred_img = []
        kernel = self.kernel_size2[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:
                padding_num = (K//2) * rate
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])
                for i in range(0, K):
                    for j in range(0, K):
                        img_stack.append(frame_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
                img_stack = torch.stack(img_stack, dim=2)
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
