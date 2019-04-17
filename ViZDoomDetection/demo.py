"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: demo.py
"""
import init_dirs
import numpy as np
import tensorflow as tf
from model import DetectionModel
from dataflow import image_height, image_width, RPN_Test
from TART.utils import check_file, get_filename
from PIL import Image
from TART.model import Saver
from TARTDetection.utils import IMGLIB

r_split = 3
c_split = 6
r_h = int(image_height / r_split)
c_w = int(image_width / c_split)

def label2img(label):
    outputs = np.zeros((image_height, image_width, 3))
    for r in range(r_split):
        for c in range(c_split):
            outputs[r * r_h:(r + 1) * r_h, c * c_w:(c + 1) * c_w, :] = np.clip(
                label[r * c_split + c], 0,
                255)
    return outputs.astype(np.uint8)


if __name__ == '__main__':

    model_path = './weights/detection.pkl'


    check_file(model_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = Saver(sess)
    image = tf.placeholder(tf.float32, [1, image_height, image_width, 3])
    model = DetectionModel()
    model.build(image)
    sess.run(tf.global_variables_initializer())
    saver.load(model_path, model=model, strict=True)

    images = ['./images/1.jpg']
    testDeal = RPN_Test()
    imglib = IMGLIB()
    for image_path in images:
        im = Image.open(image_path)
        pix = np.array(im.getdata()).reshape(1, image_height, image_width, 3).astype(np.float32)
        [test_prob, test_bbox_pred] = sess.run([model.prob, model.bbox_pred], feed_dict={image: pix})
        bbox = testDeal.rpn_nms_v2(test_prob, test_bbox_pred, thr=0.998)
        imglib.img = im
        imglib.setBBXs(bbox)
        imglib.drawBox()
        imglib.save_img('./images/' + get_filename(image_path) + '_prediction.jpg')
