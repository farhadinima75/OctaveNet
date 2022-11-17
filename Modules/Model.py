import torch, torch.nn as nn, torch.nn.functional as F, os, glob, shutil
import numpy as np, sys, tqdm, cv2, time, gc, random
from torchvision import transforms as T
from torch.nn import init
from torch import Tensor
import torchvision.transforms.functional as TF, random, pytz, datetime

from Modules.Resnet import resnet18

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>

class Layer_norm(nn.Module):
    def __init__(self, normShape, eps=1e-6, input_format="Channel_Last"):
        super(Layer_norm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normShape))
        self.bias = nn.Parameter(torch.zeros(normShape))
        self.eps = eps
        self.dataFormat = input_format
        if self.dataFormat not in ["Channel_Last", "Channel_First"]:
            raise NotImplementedError
        self.normShape = (normShape, )

    def forward(self, x):
        if self.dataFormat == "Channel_Last":
            return F.layer_norm(x, self.normShape, self.weight, self.bias, self.eps)
        elif self.dataFormat == "Channel_First":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
          
class ConvNorm(nn.Module):
    def __init__(self, DimIn, DimOut, KernelSize, Padding):
      super(ConvNorm, self).__init__()
      self.ConvNorm0 = nn.Sequential(Layer_norm(DimIn, eps=1e-6, input_format="Channel_First"),
                                     nn.Conv2d(DimIn, DimOut, kernel_size=KernelSize, stride=1, padding=Padding))
    def forward(self, x): return self.ConvNorm0(x)

class DWConv(nn.Module):
    def __init__(self, DimIn, DimOut, KernelSize, Padding):
      super(DWConv, self).__init__()
      self.ConvNorm = nn.Sequential(Layer_norm(DimIn, eps=1e-6, input_format="Channel_First"),
                                     nn.Conv2d(DimIn, DimOut, kernel_size=KernelSize, stride=1, padding=Padding, groups=DimIn))
    def forward(self, x): return self.ConvNorm(x)

class PWConv(nn.Module):
    def __init__(self, DimIn, DimOut, KernelSize, Padding):
      super(PWConv, self).__init__()
      self.ConvNorm = nn.Sequential(Layer_norm(DimIn, eps=1e-6, input_format="Channel_First"),
                                     nn.Conv2d(DimIn, DimOut, kernel_size=KernelSize, stride=1, padding=Padding))
    def forward(self, x): return self.ConvNorm(x)

class _ResNet_(torch.nn.Module):
    def __init__(self, InCH=3, resnet_stages_num=5, backbone='resnet18'):
        super(_ResNet_, self).__init__()
        expand = 1
        if InCH != 3: PreTrained = False
        else: PreTrained = True
        if backbone == 'resnet18':
            self.resnet = resnet18(pretrained=True, InCH=InCH, replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet34':
            self.resnet = resnet18(pretrained=True, InCH=InCH, replace_stride_with_dilation=[False,True,True])
        self.resnet_stages_num = resnet_stages_num
        # self.conv_pred = nn.Conv2d(512, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.forward_single(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        x_8 = self.resnet.layer3(x_8) # 1/8, in=128, out=256
        # x_8 = self.resnet.layer4(x_8) # 1/32, in=256, out=512
        # x_8 = self.conv_pred(x_8)
        return x_8

class MSTF(nn.Module):
    def __init__(self, In_Channels, Out_Channels):
      super(MSTF, self).__init__()
      # self.TripletAtt = TripletAttention()
      self.DWC = DWConv(In_Channels, In_Channels, 3, 1)
      self.PWC = PWConv(In_Channels, In_Channels, 1, 0)
      self.Conv = ConvNorm(In_Channels, In_Channels, 3, 1)
      self.LastConv = ConvNorm(In_Channels*3, Out_Channels, 3, 1)
    def forward(self, X):
      # X = self.TripletAtt(X)
      X1 = self.DWC(X)
      X2 = self.PWC(X)
      X3 = self.Conv(X)
      return self.LastConv(torch.cat([X1, X2, X3], dim=1))

class Decoder(nn.Module):
    def __init__(self, Dims, Classes):
      super(Decoder, self).__init__()
      self.FirstConv = ConvNorm(256, Dims[0], 3, 1)
      self.MSTF0_1 = MSTF(Dims[0], Dims[1])
      self.MSTF00 = MSTF(Dims[0], Dims[1])
      self.MSTF01 = MSTF(Dims[0], Dims[1])
      self.MSTF02 = MSTF(Dims[0], Dims[1])

      self.MSTF10 = MSTF(Dims[1], Dims[2])
      self.MSTF11 = MSTF(Dims[1], Dims[2])

      self.MSTF20 = MSTF(Dims[2], Dims[3])
      
      self.LastConv = ConvNorm(Dims[3], Classes, 3, 1)

    def forward(self, X):
      X = self.FirstConv(X)
      x0_1 = self.MSTF0_1(F.interpolate(X, (X.shape[-1]//2, X.shape[-2]//2), mode='bilinear'))
      x00 = self.MSTF00(X) 
      x01 = self.MSTF01(F.interpolate(X, (X.shape[-1]*2, X.shape[-2]*2), mode='bilinear')) 
      x02 = self.MSTF02(F.interpolate(X, (X.shape[-1]*4, X.shape[-2]*4), mode='bilinear')) 

      x10 = self.MSTF10(x00 + F.interpolate(x01, (x00.shape[-1], x00.shape[-2]), mode='bilinear') +\
                              F.interpolate(x02, (x00.shape[-1], x00.shape[-2]), mode='bilinear') +\
                              F.interpolate(x0_1, (x00.shape[-1], x00.shape[-2]), mode='bilinear'))
      x11 = self.MSTF11(x01 + F.interpolate(x00, (x01.shape[-1], x01.shape[-2]), mode='bilinear') +\
                              F.interpolate(x02, (x01.shape[-1], x01.shape[-2]), mode='bilinear') +\
                              F.interpolate(x0_1, (x01.shape[-1], x01.shape[-2]), mode='bilinear'))
      
      XA = self.MSTF20(x11 + F.interpolate(x10, (x11.shape[-1], x11.shape[-2]), mode='bilinear'))
      XA = F.interpolate(XA, (XA.shape[-1]*4, XA.shape[-2]*4), mode='bilinear')
      # XA = torch.sigmoid(XA)
      XA = self.LastConv(XA)
      return XA

class OctaveNet(nn.Module):
    def __init__(self, Dims, InCH=3, Classes=2):
      super(OctaveNet, self).__init__()
      #Dims could be [64, 32, 16, 16], [32, 16, 8, 8] or [8, 8, 8, 8]
      self.NetA = _ResNet_(InCH=InCH)
      self.NetB = _ResNet_(InCH=InCH)
      self.Decoder = Decoder(Dims, Classes)
      init_weights(self.Decoder)

    def forward(self, xa, xb):
      xa = self.NetA(xa)
      xb = self.NetB(xb)
      x = self.Decoder(torch.abs(xb - xa))
      return x
