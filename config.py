#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:09
#
# @Author: Greg Gao(laygin)
#'''
import os

# base_dir = 'path to dataset base dir'
base_dir = '/home/like/data/VOC'
img_dir = os.path.join(base_dir, 'VOC2007/JPEGImages')
xml_dir = os.path.join(base_dir, 'VOC2007/Annotations')

train_txt_file = os.path.join(base_dir, r'VOC2007/ImageSets/Main/train.txt')
val_txt_file = os.path.join(base_dir, r'VOC2007/ImageSets/Main/val.txt')


anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 500
RPN_TOTAL_NUM = 1000

# bgr can find from  here: https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68, 116.779, 103.939]
IMAGE_MEAN_NEW=[127.5]

checkpoints_dir = '/home/like/data/checkpoints'
outputs = r'./logs'
