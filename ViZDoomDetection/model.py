"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: model.py
"""
import init_dirs

import numpy as np
import tensorflow as tf
import dataflow
from TART.model import Model

classNames = dataflow.loadClass()
VGG_MEAN = [103.939, 116.779, 123.68]
image_height = dataflow.image_height
image_width = dataflow.image_width
feature_height = int(np.ceil(image_height / dataflow.featureReduce))
feature_width = int(np.ceil(image_width / dataflow.featureReduce))


class DetectionModel(Model):
    def build(self, rgb):
        red, green, blue = tf.split(rgb, 3, 3)
        assert red.get_shape().as_list()[1:] == [image_height, image_width, 1]
        assert green.get_shape().as_list()[1:] == [image_height, image_width, 1]
        assert blue.get_shape().as_list()[1:] == [image_height, image_width, 1]
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], 3)
        assert bgr.get_shape().as_list()[1:] == [image_height, image_width, 3]
        self.conv1_1 = self.conv2d(bgr, 'conv1_1', out_channel=64, use_relu=True)
        self.pool1 = self.max_pool(self.conv1_1, 'pool1')
        self.conv2_1 = self.conv2d(self.pool1, 'conv2_1', out_channel=128, use_relu=True)
        self.pool2 = self.max_pool(self.conv2_1, 'pool2')
        self.conv3_1 = self.conv2d(self.pool2, 'conv3_1', out_channel=256, use_relu=True)
        self.pool3 = self.max_pool(self.conv3_1, 'pool3')
        self.conv4_1 = self.conv2d(self.pool3, 'conv4_1', out_channel=512, use_relu=True)
        self.pool4 = self.max_pool(self.conv4_1, 'pool4')
        self.conv5_1 = self.conv2d(self.pool4, 'conv5_1', out_channel=512, use_relu=True)
        self.relu_proposal_all = self.conv5_1
        self.conv_cls_score = self.conv2d(self.relu_proposal_all, 'conv_cls_score', out_channel=18, kernel_size=[1, 1],
                                          use_relu=False)
        self.conv_bbox_pred = self.conv2d(self.relu_proposal_all, 'conv_bbox_pred', out_channel=36, kernel_size=[1, 1],
                                          use_relu=False)
        assert self.conv_cls_score.get_shape().as_list()[1:] == [feature_height, feature_width,
                                                                 9 * (dataflow.classNumber + 1)]
        assert self.conv_bbox_pred.get_shape().as_list()[1:] == [feature_height, feature_width, 36]
        self.cls_score = tf.reshape(self.conv_cls_score, [-1, dataflow.classNumber + 1])
        self.bbox_pred = tf.reshape(self.conv_bbox_pred, [-1, 4])
        self.prob = tf.nn.softmax(self.cls_score, name="prob")
