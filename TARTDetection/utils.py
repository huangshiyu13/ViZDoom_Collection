#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shiyu Huang
# @File    : utils.py

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
from TART.utils import get_all_files
import pickle
from sklearn.utils.extmath import cartesian


def IOU(x, centroids):
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            similarity = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            similarity = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w * c_h) / (w * h)
        similarities.append(similarity)  # will become (k,) shape
    return np.array(similarities)


def kmeans(annotations, centroids):
    anno_number = annotations.shape[0]
    anchor_number, dim = centroids.shape

    prev_assignments = np.ones(anno_number) * (-1)

    iter = 0
    while True:
        D = []
        iter += 1
        for i in range(anno_number):
            d = 1 - IOU(annotations[i], centroids)
            D.append(d)
        D = np.array(D)

        assignments = np.argmin(D, axis=1)
        if (assignments == prev_assignments).all():
            print(anno_number, [np.sum(assignments == j) for j in range(anchor_number)])
            print(centroids)
            return centroids

        centroid_sums = np.zeros((anchor_number, dim), np.float)
        for i in range(anno_number):
            centroid_sums[assignments[i]] += annotations[i]
        for j in range(anchor_number):

            if np.sum(assignments == j) != 0:
                centroids[j] = centroid_sums[j] / (np.sum(assignments == j))
            else:
                centroids[j] = annotations[random.randint(0, len(annotations) - 1)]

        prev_assignments = assignments.copy()


def gen_anchors(anno_dir, class_names, save_path, max_size, anchor_number=9):
    files = get_all_files(anno_dir, '.txt')

    class_wh = {}
    for name in class_names:
        class_wh[name] = []

    for file in files:
        f = open(file, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line_s = line.split(' ')
            class_wh[line_s[0]].append([float(line_s[3]), float(line_s[4])])

    miss_key = []

    anchor_dict = {}
    for key in class_wh:
        if len(class_wh[key]) == 0:
            miss_key.append(key)
            continue
        print('{}:'.format(key))
        annotations = np.array(class_wh[key])
        indices = [random.randrange(annotations.shape[0]) for _ in range(anchor_number)]
        centroids = annotations[indices]  # centroids中是该类的annotations中随机的anchor_number个编号
        centroids = kmeans(annotations, centroids)

        centroids = np.minimum(centroids, max_size)

        anchors = centroids.copy()

        widths = anchors[:, 0]
        sorted_indices = np.argsort(widths)
        anchor_dict[key] = anchors[sorted_indices]

    for key in miss_key:
        anchor_dict[key] = anchors[sorted_indices]
    with open(save_path, 'wb') as f:
        pickle.dump(anchor_dict, f, pickle.HIGHEST_PROTOCOL)


def safeInt(ss):
    return int(float(ss))


def get_region(im):
    size_x, size_y = im.shape[0], im.shape[1]
    x_min, x_max = -1, size_x
    y_min, y_max = -1, size_y
    for i in range(2, size_x):
        if np.mean(im[i, :]) > 30:
            x_min = i
            break
    for i in range(2, size_x):
        if np.mean(im[size_x - i, :]) > 30:
            x_max = size_x - i
            break
    for i in range(2, size_y):
        if np.mean(im[:, i]) > 30:
            y_min = i
            break
    for i in range(2, size_y):
        if np.mean(im[:, size_y - i]) > 30:
            y_max = size_y - i
            break
    return x_min, x_max, y_min, y_max


class BBX:
    def __init__(self, x=None, y=None, w=None, h=None, score=None, name=None):
        self.name = name

        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score

    def str2bbx(self, str):
        chrs = str.split(' ')

        self.name = chrs[0]

        self.x = safeInt(chrs[1])
        self.y = safeInt(chrs[2])
        self.w = safeInt(chrs[3])
        self.h = safeInt(chrs[4])
        self.score = float(chrs[5])

    def str2bbx_true(self, str):
        chrs = str.split(' ')

        self.name = chrs[0]

        self.x = safeInt(chrs[1])
        self.y = safeInt(chrs[2])
        self.w = safeInt(chrs[3])
        self.h = safeInt(chrs[4])
        self.score = 1

    def resize(self, scale, x_d, y_d):
        self.x = safeInt(self.x * scale) + x_d
        self.y = safeInt(self.y * scale) + y_d
        self.w = safeInt(self.w * scale)
        self.h = safeInt(self.h * scale)


class COLOR_CONF:
    def __init__(self, names=[], default_color=(255, 0, 0), default_font_size=12, line_width=1):
        self.colors = {}
        self.names = names
        if names is not None:
            self.generate_colors(names)

        self.default_color = default_color
        self.default_font_size = default_font_size
        self.line_width = line_width

    def set_color(self, name, color):
        self.colors[name] = color

    def generate_colors(self, names):
        for i in range(len(names)):
            self.colors[names[i]] = (random.randint(0, 125), random.randint(0, 125), random.randint(0, 125))

    def get_color(self, name):
        if name in self.colors:
            return self.colors[name]
        else:
            return self.default_color


class IMGLIB:
    def __init__(self, color_conf=None):
        if color_conf is None:
            default_color = (255, 0, 0)
            self.color_conf = COLOR_CONF(default_color=default_color)
        else:
            self.color_conf = color_conf

        FontData = os.path.join(os.path.dirname(os.path.realpath(__file__)), "OpenSans-Regular.ttf")
        self.font = ImageFont.truetype(FontData, self.color_conf.default_font_size)

    def setBBXs(self, bboxs=None, names=None):
        self.bbxs = []
        for i, bbox in enumerate(bboxs):

            bbx = BBX()

            if names == None:
                bbx.name = None
            else:
                bbx.name = names[i]
            bbx.x = safeInt(bbox[0])
            bbx.y = safeInt(bbox[1])
            bbx.w = safeInt(bbox[2])
            bbx.h = safeInt(bbox[3])
            bbx.score = bbox[4]
            self.bbxs.append(bbx)

    def setTrueBBXs(self, bboxs=None, names=None):
        self.bbxs_true = []
        for i, bbox in enumerate(bboxs):

            bbx = BBX()

            if names is None:
                bbx.name = None
            else:
                bbx.name = names[i]
            bbx.x = safeInt(bbox[0])
            bbx.y = safeInt(bbox[1])
            bbx.w = safeInt(bbox[2])
            bbx.h = safeInt(bbox[3])
            bbx.score = 1.
            self.bbxs_true.append(bbx)

    def showBBXs(self):
        self.drawBox()
        self.img.show()

    def saveBBXs(self, fileName):
        f = open(fileName, 'w')
        for bbx in self.bbxs:
            f.write('%s %d %d %d %d %f\n' % (bbx.name, bbx.x, bbx.y, bbx.w, bbx.h, bbx.score))
        f.close()

    def drawOneBox(self, bbx, thr=-1.0, showName=False):
        if bbx.score >= thr:

            x = bbx.x
            y = bbx.y
            w = bbx.w
            h = bbx.h
            # print x,y,w,h
            line1 = ((x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y))

            fill_color = self.color_conf.get_color(bbx.name)
            # print line1
            # print(fill_color)
            # print self.color_conf.line_width
            self.draw.line(line1, fill=fill_color, width=self.color_conf.line_width)

            if bbx.name == None or showName == False:
                self.draw.text((x + self.color_conf.line_width, y), str(bbx.score), fill=fill_color, font=self.font)
            else:
                self.draw.text((x + self.color_conf.line_width, y), bbx.name + ' ' + str(bbx.score), fill=fill_color,
                               font=self.font)

    def drawOneBoxTrue(self, bbx, showName=False):
        x = bbx.x
        y = bbx.y
        w = bbx.w
        h = bbx.h
        line1 = ((x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y))
        fill_color = self.color_conf.get_color(bbx.name)
        self.draw.line(line1, fill=fill_color, width=self.color_conf.line_width)

        if bbx.name == None or showName == False:
            self.draw.text((x + self.color_conf.line_width, y), 'True', fill=fill_color, font=self.font)
        else:

            self.draw.text((x + self.color_conf.line_width, y), bbx.name + '_True', fill=fill_color, font=self.font)

    def drawBox(self, thr=-1.0, showName=True, show_true=True, strict=False):
        self.draw = ImageDraw.Draw(self.img)

        if hasattr(self, 'bbxs'):
            for bbx in self.bbxs:
                if strict and bbx.name not in self.color_conf.names:
                    continue
                self.drawOneBox(bbx, thr, showName)

        if show_true and hasattr(self, 'bbxs_true'):
            for bbx in self.bbxs_true:
                if strict and bbx.name not in self.color_conf.names:
                    continue
                self.drawOneBoxTrue(bbx, showName)

    def read_img(self, fileName):
        self.img = Image.open(fileName).convert('RGB')

    def read_gray_img(self, fileName):
        self.img = Image.open(fileName).convert('L')

    def read_ano(self, fileName):

        f = open(fileName, 'r')
        lines = f.readlines()
        self.bbxs = []
        for line in lines[:]:
            nbbx = BBX()
            nbbx.str2bbx(line)
            self.bbxs.append(nbbx)

    def read_ano_true(self, fileName):

        f = open(fileName, 'r')
        lines = f.readlines()
        self.bbxs_true = []
        for line in lines[:]:
            nbbx = BBX()
            nbbx.str2bbx_true(line)
            self.bbxs_true.append(nbbx)

    def resizeBBXs(self, r, x_d, y_d):
        for bbx in self.bbxs:
            bbx.resize(r, x_d, y_d)

    def resize(self, width, height, scale=1.0):
        o_width, o_height = self.img.size
        t_width = safeInt(width * scale)
        t_height = safeInt(height * scale)

        o_ratio = o_width / float(o_height)
        n_ratio = width / float(height)

        if o_ratio > n_ratio:
            re_ration = t_width / float(o_width)
            a_height = safeInt(re_ration * o_height)
            a_width = t_width
            self.img = self.img.resize((a_width, a_height), Image.ANTIALIAS)
        else:
            re_ration = t_height / float(o_height)
            a_width = safeInt(re_ration * o_width)
            a_height = t_height
            self.img = self.img.resize((a_width, a_height), Image.ANTIALIAS)

        self.x_d = random.randint(0, abs(a_width - width))
        self.y_d = random.randint(0, abs(a_height - height))
        imgNew = Image.new("RGB", (width, height), "black")

        box = (0, 0, a_width, a_height)
        region = self.img.crop(box)

        imgNew.paste(region, (self.x_d, self.y_d))
        self.img = imgNew
        if hasattr(self, 'bbxs'):
            self.resizeBBXs(re_ration, self.x_d, self.y_d)
        # self.drawBox()

    def cleanAno(self, w0, h0):
        newBBXS = []
        for bbox in self.bbxs:
            if bbox.x >= 0 and bbox.x <= w0 and bbox.y >= 0 and bbox.y <= h0 and bbox.w >= 20 and bbox.w <= w0 and bbox.h >= 30 and bbox.h <= h0:
                bbx = BBX()
                bbx.name = bbox.name
                bbx.x = bbox.x
                bbx.y = bbox.y
                bbx.w = bbox.w
                bbx.h = bbox.h
                bbx.score = bbox.score
                newBBXS.append(bbx)
        self.bbxs = newBBXS

    def save_img(self, imgName):
        self.img.save(imgName)

    def pureResize(self, width, height):
        re_ration = float(width) / self.img.size[0]
        self.img = self.img.resize((width, height), Image.ANTIALIAS)
        if hasattr(self, 'bbxs'):
            self.resizeBBXs(re_ration, 0, 0)

    def pureResizeBBX(self, original_width, width, height):
        re_ration = float(width) / original_width
        if hasattr(self, 'bbxs'):
            self.resizeBBXs(re_ration, 0, 0)

    def flip(self, width):
        self.img = self.img.transpose(Image.FLIP_LEFT_RIGHT)
        newBBXS = []
        for bbox in self.bbxs:
            bbox.x = width - bbox.x - bbox.w
            newBBXS.append(bbox)
        self.bbxs = newBBXS

    def normalization(self, mean_pix):  # 除以均值
        im = np.asarray(self.img, dtype=np.float)
        x_min, x_max, y_min, y_max = get_region(im)
        tmp = im[im > 20]
        if len(tmp) == 0:
            return
        im *= mean_pix / np.mean(tmp)
        # delta = np.mean(im[x_min:x_max, y_min:y_max]) - mean_pix
        # im[x_min:x_max, y_min:y_max] -= delta
        im[im < 0] = 0
        im[im > 255] = 255
        self.img = Image.fromarray(np.uint8(im))

    # def normalization(self, mean_pix, var_pix):  #减均值
    #     im = np.asarray(self.img, dtype=np.float)
    #     x_min, x_max, y_min, y_max = get_region(im)
    #     tmp = im[im > 20]
    #     if len(tmp) == 0:
    #         return
    #     im = im - np.mean(tmp) + mean_pix
    #     im[im < 0] = 0
    #     im[im > 255] = 255
    #     self.img = Image.fromarray(np.uint8(im))

    # def normalization(self, mean_pix, var_pix):  #减均值，除方差
    #     im = np.asarray(self.img, dtype=np.float)
    #     x_min, x_max, y_min, y_max = get_region(im)
    #     tmp = im[im > 20]
    #     if len(tmp) == 0:
    #         return
    #     im = im - np.mean(tmp) + mean_pix
    #     im = im * var_pix / np.var(tmp)
    #     im[im < 0] = 0
    #     im[im > 255] = 255
    #     self.img = Image.fromarray(np.uint8(im))


# mat1 --> ground truth(s); mat2 --> anchors
def compute_overlap(mat1, mat2):
    s1 = mat1.shape[0]
    s2 = mat2.shape[0]
    area1 = (mat1[:, 2] - mat1[:, 0]) * (mat1[:, 3] - mat1[:, 1])
    if mat2.shape[1] == 5:
        area2 = mat2[:, 4]
    else:
        area2 = (mat2[:, 2] - mat2[:, 0]) * (mat2[:, 3] - mat2[:, 1])
    x1 = cartesian([mat1[:, 0], mat2[:, 0]])
    x1 = np.amax(x1, axis=1)
    x2 = cartesian([mat1[:, 2], mat2[:, 2]])
    x2 = np.amin(x2, axis=1)
    com_zero = np.zeros(x2.shape[0])
    w = x2 - x1
    w = w - 1
    w = np.maximum(com_zero, w)
    y1 = cartesian([mat1[:, 1], mat2[:, 1]])
    y1 = np.amax(y1, axis=1)
    y2 = cartesian([mat1[:, 3], mat2[:, 3]])
    y2 = np.amin(y2, axis=1)
    h = y2 - y1
    h = h - 1
    h = np.maximum(com_zero, h)
    oo = w * h
    aa = cartesian([area1[:], area2[:]])
    aa = np.sum(aa, axis=1)
    ooo = oo / (aa - oo)
    overlap = np.transpose(ooo.reshape(s1, s2), (1, 0))
    return overlap


# mat1 --> ground truth; mat2 --> anchor
def compute_regression(mat1, mat2):
    target = np.zeros(4)
    w1 = mat1[2] - mat1[0]
    h1 = mat1[3] - mat1[1]
    w2 = mat2[2] - mat2[0]
    h2 = mat2[3] - mat2[1]
    target[0] = (mat1[0] - mat2[0]) / w2
    target[1] = (mat1[1] - mat2[1]) / h2
    target[2] = np.log(w1 / w2)
    target[3] = np.log(h1 / h2)
    return target


def compute_target(roi_t, proposals, fg_thresh, bg_thresh):
    roi = roi_t.copy()
    roi[:, 2] += roi[:, 0]
    roi[:, 3] += roi[:, 1]
    proposal_size = proposals.shape[0]
    roi_anchor = np.zeros([proposal_size, 5])
    # roi_anchor[:, 0] = 0 #change here to set all bbx to background
    if roi.shape[0] == 0:
        # roi_anchor[:, 0] = 0
        return roi_anchor, 0
    overlap = compute_overlap(roi, proposals)
    overlap_max = np.max(overlap, axis=1)
    overlap_max_idx = np.argmax(overlap, axis=1)
    for i in range(proposal_size):
        if overlap_max[i] >= fg_thresh:
            roi_anchor[i, 0] = 1
            roi_anchor[i, 1:5] = compute_regression(roi[overlap_max_idx[i], :4], proposals[i, :])
        if overlap_max[i] <= bg_thresh:
            roi_anchor[i, 0] = 0
    foreground = np.sum(roi_anchor[:, 0] > 1)
    return roi_anchor, foreground


def compute_target_v2(roi_t, proposals, fg_thresh):
    roi = roi_t.copy()
    roi[:, 2] += roi[:, 0]
    roi[:, 3] += roi[:, 1]
    proposal_size = proposals.shape[0]
    roi_anchor = np.zeros([proposal_size, 5])
    # roi_anchor[:, 0] = 0 #change here to set all bbx to background
    if roi.shape[0] == 0:
        # roi_anchor[:, 0] = 0
        return roi_anchor, 0
    overlap = compute_overlap(roi, proposals)
    overlap_max = np.max(overlap, axis=1)
    overlap_max_idx = np.argmax(overlap, axis=1)
    for i in range(proposal_size):
        if overlap_max[i] >= fg_thresh:
            roi_anchor[i, 0] = 1
            roi_anchor[i, 1:5] = compute_regression(roi[overlap_max_idx[i], :4], proposals[i, :])
        # else:
        #     roi_anchor[i, 0] = 0
    foreground = np.sum(roi_anchor[:, 0] > 1)
    return roi_anchor, foreground
