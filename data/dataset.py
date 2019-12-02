# -*- coding:utf-8 -*-
# '''
# Created on 18-12-27 上午10:34
#
# @Author: Greg Gao(laygin)
# '''

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
from config import IMAGE_MEAN,IMAGE_MEAN_NEW
from ctpn_utils import cal_rpn
import ipdb
import math


def readxml(path, height, width):
    # 规整化成1080 X1920
    gtboxes = []
    imgfile = ''
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'filename' in elem.tag:
            imgfile = elem.text
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    xmin = float(attr.find('xmin').text)
                    ymin = float(attr.find('ymin').text)
                    xmax = float(attr.find('xmax').text)
                    ymax = float(attr.find('ymax').text)
                    #
                    # nx1=int(round((1920*xmin/width)))
                    # ny1=int(round((1080*ymin/height)))
                    # nx2=int(round((1920*xmax/width)))
                    # ny2=int(round((1080*ymax/height)))

                    gtboxes.append((xmin, ymin, xmax, ymax))

    return np.array(gtboxes), imgfile

def dealgray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(gray, kernel)
    # cv2.imwrite("1.png", eroded)
    gray = cv2.medianBlur(eroded, 3)
    # cv2.imwrite("2.png",gray)
    return gray


# for ctpn text detection
class VOCDataset(Dataset):
    def __init__(self,
                 datadir,
                 labelsdir):
        '''

        :param txtfile: image name list text file
        :param datadir: image's directory
        :param labelsdir: annotations' directory
        '''
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.datadir2="/home/like/data/VOC/VOC2007/JPEGImages2"
        # self.img_names = os.listdir(self.datadir)

        self.labelsdir = labelsdir
        # 取出文件夹中的文件
        self.img_names = []
        image_set_file = "/home/like/data/VOC/VOC2007/ImageSets/Main/trainval.txt"
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            self.img_names = [x.strip() + '.png' for x in f.readlines()]

        self.img_names2=[]
        image_set_file2 = "/home/like/data/VOC/VOC2007/ImageSets/Main/trainval2.txt"
        assert os.path.exists(image_set_file2), \
            'Path does not exist: {}'.format(image_set_file2)
        with open(image_set_file2) as f:
            self.img_names2 = [x.strip() + '.jpg' for x in f.readlines()]

        self.img_name=self.img_names+self.img_names2

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        #new work
        img_path = os.path.join(self.datadir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.datadir2, img_name)
        xml_path = ""
        lim = img_name.split('_')[0]
        if lim not in ["img","image","tainchi"]:
            xml_path = os.path.join(self.labelsdir, img_name.replace('.png', '.xml'))
        else:
            self.labelsdir2 = "/home/like/data/VOC/VOC2007/Annotations2"
            xml_path = os.path.join(self.labelsdir2, img_name.replace('.jpg', '.xml'))
        #new work end
        img = cv2.imread(img_path)

        gray = dealgray(img)
        gray=gray-IMAGE_MEAN_NEW


        height, width = img.shape[:2]
        gtbox, _ = readxml(xml_path, height, width)  # guiyihuatuxian
        # img=cv2.resize(img,(1080,1920),interpolation=cv2.INTER_CUBIC)
        h, w, c = img.shape
        # clip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], _ = cal_rpn((h, w), (math.ceil(h / 16), math.ceil(w / 16)), 16, gtbox)

        m_img = img - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        gray = torch.from_numpy(gray).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()


        return m_img, gray,cls, regr
