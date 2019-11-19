# -*- coding:utf-8 -*-
# '''
# Created on 18-12-11 上午10:01
#
# @Author: Greg Gao(laygin)
# '''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from retina_fpn import RetinaFPN101
import numpy as np
import ipdb

class CTC_ATTENTION(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(CTC_ATTENTION, self).__init__()
        self.cov1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1,bias=True),
            nn.BatchNorm2d(32,eps=0.0002),
            nn.ReLU()
        )
        self.cov2=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,bias=True),
            nn.BatchNorm2d(32,eps=0.0002),
            nn.ReLU()
        )
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
        self.cov3=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,bias=True),
            nn.BatchNorm2d(32,eps=0.0002),
            nn.ReLU()
        )
        self.cov4=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1,bias=True),
            nn.BatchNorm2d(32,eps=0.0002),
            nn.ReLU()
        )
        self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
        self.cov5=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,bias=True),
            nn.BatchNorm2d(32,eps=0.0002),
            nn.ReLU()
        )
        self.cov6=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,bias=True),
            nn.BatchNorm2d(32,eps=0.0002),
            nn.ReLU()
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.cov7=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,bias=True),
            nn.BatchNorm2d(32,eps=0.0002),
            nn.ReLU()
        )
        self.cov8=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1,bias=True),
            nn.BatchNorm2d(32,eps=0.0002),
            nn.ReLU()
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.linear1=nn.Sequential(
            nn.Linear(in_features=10000,out_features=512),
            nn.ReLU(),
            nn.BatchNorm2d(eps=0.0002),
            nn.Dropout(0.2)
        )
        self.linear2=nn.Sequential(
            nn.Linear(in_features=10000,out_features=512),
            nn.ReLU(),
        )

        self.attention_prob=nn.Sequential(
            nn.Linear(1230,234124),
            nn.Softmax(),
        )

        self.dense1_bn=nn.BatchNorm2d(eps=0.0002)
        self.dense1_dropout=nn.Dropout(0.3)


        self.dense2=nn.Linear(in_features=512,out_features=123213)
        self.dense2_ac=nn.Softmax()


    def forward(self,x):
        x1=self.cov1(x)
        x2=self.cov2(x1)
        x3=self.maxpool1(x2)
        x4=self.cov3(x3)
        x5=self.cov4(x4)
        x6=self.maxpool2(x5)
        x7=self.cov5(x6)
        x8=self.cov6(x7)
        x9=self.maxpool3(x8)
        x10=self.cov7(x9)
        x11=self.cov8(x10)
        x12=self.maxpool4(x11)



        x2=self.cov2(x1)



