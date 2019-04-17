"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: model.py
"""

import tensorflow as tf
from TART.model import Model

VGG_MEAN = [103.939, 116.779, 123.68]

class DepthModel(Model):
    def build(self, rgbs):
        self.img = rgbs
        red, green, blue = tf.split(rgbs, 3, 3)
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], 3)
        self.conv1_1 = self.conv2d(bgr, 'conv1_1',out_channel=32,use_relu=True)
        self.pool1 = self.max_pool(self.conv1_1, 'pool1')
        self.conv2_1 = self.conv2d(self.pool1, 'conv2_1',out_channel=16,use_relu=True)
        self.pool2 = self.max_pool(self.conv2_1, 'pool2')
        self.conv3_1 = self.conv2d(self.pool2, 'conv3_1',out_channel=8,use_relu=True)
        self.pool3 = self.max_pool(self.conv3_1, 'pool3')
        self.fc1 = self.dense(self.pool3, 'fc1', in_size=4608, out_size=128, use_relu=True)
        self.fc2 = self.dense(self.fc1, 'fc2', in_size=128, out_size=18)
        self.predict = self.fc2
