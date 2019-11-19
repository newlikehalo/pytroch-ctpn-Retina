#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:03
#
# @Author: Greg Gao(laygin)
#'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cv2
import numpy as np
import glob
import torch
import torch.nn.functional as F
from ctpn_model import CTPN_Model
from ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox,nms, TextProposalConnectorOriented
from ctpn_utils import resize
import config
import ipdb
import time
import math


def cutstr(string):
    return string.split('/')[-1].split('.')[0]
def ifdir(dir):  #判断是不是有这个目录
    if not os.path.exists(dir):
        os.mkdir(dir)
ALL_DIR="/home/like/data/ctpnresult"
EPOCH="epoch_10"

EPOCH_DIR=os.path.join(ALL_DIR,EPOCH)

newepoch=os.path.join(EPOCH_DIR,str(config.IOU_SELECT))
ifdir(newepoch)
EPOCH_IMAGE=os.path.join(newepoch,"imageresult")
EPOCH_TXT=os.path.join(newepoch,"pthfile")

ifdir(EPOCH_DIR)
ifdir(EPOCH_IMAGE)
ifdir(EPOCH_TXT)


prob_thresh =config.IOU_SELECT
width = 600
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weights = os.path.join(EPOCH_DIR, 'ctpn_ep09_0.0054_0.0045_0.0099.pth.tar')
# img_path = '/home/like/data/pic/image/without_label/37.JPG'

model = CTPN_Model()
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
model.to(device)
model.eval()

def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test(img_path):
    image = cv2.imread(img_path)
    # ipdb.set_trace()
    image_c = image.copy()
    # h1,w1,c=image_c.shape
    # oddnumber=w1/width
    # image = resize(image, width=width)
    h, w = image.shape[:2]
    image = image.astype(np.float32) - config.IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        image = image.to(device)
        cls, regr = model(image)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()
        anchor = gen_anchor((math.ceil(h / 16), math.ceil(w / 16)), 16)
        bbox = bbox_transfor_inv(anchor, regr)
        bbox = clip_box(bbox, [h, w])
        fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)
        keep_index = filter_bbox(select_anchor, 16)

        # nsm
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, 0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]
        # text line-
        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])
        # print(text)
        alltext=[]
        for i in text:
            if i[-1]>=0.9:
                s = str(round(i[-1] * 100, 2)) + '%'
                i = [int(j) for j in i]
                cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2)
                cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 2)
                cv2.putText(image_c, s, (i[0] + 13, i[1] + 13),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA)

        imagename=cutstr(img_path)
        savepicpath=os.path.join(EPOCH_IMAGE,imagename+'.png')
        cv2.imwrite(savepicpath, image_c)
        save_path = os.path.join(EPOCH_TXT,imagename+'.txt')
        file = open(save_path, 'w')
        file.write(str(alltext))
        file.close()



if __name__=='__main__':
    DATA_DIR="/home/like/data/ctpnresult/testdata/pic"
    im_names = glob.glob(os.path.join(DATA_DIR, '*.png')) + \
               glob.glob(os.path.join(DATA_DIR,  '*.jpg'))
    im_names.sort()
    start=time.time()
    for im_name in im_names:
        print(im_name)
        test(im_name)
        # ctpn(sess, net, im_name)
    end=time.time()
    print(end-start)
