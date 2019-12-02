# -*- coding:utf-8 -*-
# '''
# Created on 18-12-11 上午10:03
#
# @Author: Greg Gao(laygin)
# '''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cv2
import numpy as np
import glob
import torch
import torch.nn.functional as F
from ctpn_model import CTPN_Model
from ctpn_utils import gen_anchor, bbox_transfor_inv, nms, clip_box, filter_bbox, TextProposalConnectorOriented
from ctpn_utils import resize
import config
import ipdb
import time
import math
from lib.text_proposal_connector import TextProposalConnector
import copy


# from lib.fast_rcnn.nms_wrapper import nms
def cutstr(string):
    return string.split('/')[-1].split('.')[0]


def ifdir(dir):  # 判断是不是有这个目录
    if not os.path.exists(dir):
        os.mkdir(dir)


ALL_DIR = "/home/like/data/ctpnresult"
EPOCH = "epoch_12_b"

EPOCH_DIR = os.path.join(ALL_DIR, EPOCH)

newepoch = os.path.join(EPOCH_DIR, str(config.IOU_SELECT))
ifdir(newepoch)
EPOCH_IMAGE = os.path.join(newepoch, "imageresult")
EPOCH_TXT = os.path.join(newepoch, "pthfile")

ifdir(EPOCH_DIR)
ifdir(EPOCH_IMAGE)
ifdir(EPOCH_TXT)

prob_thresh = config.IOU_SELECT
width = 600
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weights = glob.glob(os.path.join(EPOCH_DIR, '*.tar'))[0]
# torch.save(weights,weights)
ipdb.set_trace()
# img_path = '/home/like/data/pic/image/without_label/37.JPG'
# weights = "/home/like/pytorch_ctpn/checkpoints/ctpn_ep02_0.0727_0.0568_0.1295.pth.tar"
ipdb.set_trace()
model = CTPN_Model()
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
model.to(device)
model.eval()


def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# new work
def anaiou(line, inds):
    boxs = []

    for i in inds:
        bbox = line[i, :4]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        boxs.append(bbox)
    newbox = copy.deepcopy(boxs)
    for i, mbox in enumerate(boxs):
        for j, nbox in enumerate(boxs):
            if i != j:

                marea = (mbox[2] - mbox[0]) * (mbox[3] - mbox[1])
                narea = (nbox[2] - nbox[0]) * (nbox[3] - nbox[1])
                # print(mbox,nbox,marea, narea)
                x1 = max(mbox[0], nbox[0])
                x2 = min(mbox[2], nbox[2])
                y1 = max(mbox[1], nbox[1])
                y2 = min(mbox[3], nbox[3])
                intersection = max(x2 - x1, 0) * max(y2 - y1, 0)

                if intersection / marea > 0.7:
                    bx1 = min(mbox[0], nbox[0])
                    bx2 = max(mbox[2], nbox[2])
                    by1 = min(mbox[1], nbox[1])
                    by2 = max(mbox[3], nbox[3])
                    newbox[i] = [0, 0, 0, 0]
                    newbox[j] = [bx1, by1, bx2, by2]
                elif intersection / narea > 0.7:
                    bx1 = min(mbox[0], nbox[0])
                    bx2 = max(mbox[2], nbox[2])
                    by1 = min(mbox[1], nbox[1])
                    by2 = max(mbox[3], nbox[3])
                    newbox[j] = [0, 0, 0, 0]
                    newbox[i] = [bx1, by1, bx2, by2]
    nnbox = []
    for i in newbox:
        if not (i[0] == 0 and i[1] == 0 and i[2] == 0 and i[3] == 0):
            # print(i)
            nnbox.append(i)
        else:
            print("1<i")
    # ipdb.set_trace()
    return nnbox


def save_results(image_name, line, thresh):
    im = cv2.imread(image_name)
    inds = np.where(line[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    newimage_name = image_name.split('/')[-1].split('.')[0]
    all_list = []
    nnbox = anaiou(line, inds)
    for bbox in nnbox:
        # bbox = line[i, :4]
        # score = line[i, -1]
        cv2.rectangle(
            im, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
            color=(0, 0, 255),
            thickness=1)
        all_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
    save_path = os.path.join(EPOCH_TXT, newimage_name + '.txt')
    file = open(save_path, 'w')
    file.write(str(all_list))
    file.close()

    image_name = image_name.split('/')[-1]
    cv2.imwrite(os.path.join(EPOCH_IMAGE, image_name), im)


def connect_proposal(text_proposals, scores, im_size):
    cp = TextProposalConnector()
    line = cp.get_text_lines(text_proposals, scores, im_size)
    return line


def test(img_path):
    image = cv2.imread(img_path)

    """gray"""



    isize = image.shape
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
        boxes = bbox[fg, :]  # 可用的框格
        scores = cls_prob[0, fg, 1]

        select_anchor = boxes.astype(np.float32)

        keep_index = filter_bbox(select_anchor, 16)

        # nsm
        boxes = select_anchor[keep_index]
        scores = scores[keep_index]

        NMS_THRESH = 0.3
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        keep = np.where(dets[:, 4] >= 0.7)[0]
        dets = dets[keep, :]
        line = connect_proposal(dets[:, 0:4], dets[:, 4], isize)
        save_results(img_path, line, thresh=0.9)


if __name__ == '__main__':
    DATA_DIR = "/home/like/data/ctpnresult/testdata/pic"
    im_names = glob.glob(os.path.join(DATA_DIR, '*.png')) + \
               glob.glob(os.path.join(DATA_DIR, '*.jpg'))
    im_names.sort()
    start = time.time()
    # im_names=["/home/like/data/ctpnresult/testdata/111.png"]
    for im_name in im_names:
        print(im_name)
        test(im_name)
        # ctpn(sess, net, im_name)
    end = time.time()
    print(end - start)
