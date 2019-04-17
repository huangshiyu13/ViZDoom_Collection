import numpy as np
from TARTDetection import NMS

featureReduce = 16
image_height = 576
image_width = 1024
classNumber = 1
# for v9
wandhG = [[20.0, 20.0], [20.0, 40.0], [40.0, 40.0],
          [40.0, 80.0], [80.0, 80.0], [80.0, 160.0],
          [160.0, 160.0], [160.0, 320.0], [320.0, 320.0]]


def loadClass():
    names = ['other', 'enemy']
    return names


classDic = {'enemy': 1}


class RPN_Test(object):
    def __init__(self):
        self.prefilt = 0.3
        self.image_height = image_height
        self.image_width = image_width
        self.convmap_height = int(np.ceil(self.image_height / featureReduce))
        self.convmap_width = int(np.ceil(self.image_width / featureReduce))
        self.anchor_size = 9
        self.bbox_normalize_scale = 5
        self.wandh = wandhG
        self.proposal_prepare()

    def rpn_nms_v2(self, prob, bbox_pred, thr):
        prob2 = prob[:, 1:classNumber + 1].max(axis=1)
        index = np.where(prob2 >= thr)[0]
        if len(index) == 0:
            return []
        bbox_pred /= self.bbox_normalize_scale
        anchors = self.proposals.copy()
        bbox_pred = bbox_pred[index]
        anchors = anchors[index]
        prob_class = np.argmax(prob[index, 1:classNumber + 1], axis=1) + 1
        prob = prob2[index]
        anchors[:, 0] = bbox_pred[:, 0] * anchors[:, 2] + anchors[:, 0]
        anchors[:, 1] = bbox_pred[:, 1] * anchors[:, 3] + anchors[:, 1]
        anchors[:, 2] = np.exp(bbox_pred[:, 2]) * anchors[:, 2]
        anchors[:, 3] = np.exp(bbox_pred[:, 3]) * anchors[:, 3]
        bbox = np.zeros([anchors.shape[0], 6])
        bbox[:, :4] = anchors
        bbox[:, 4] = prob
        bbox[:, 5] = prob_class
        bbox = NMS.filter_bbox(bbox, image_width, image_height)
        bbox = NMS.non_max_suppression_fast(bbox, 0.4)
        return bbox

    def proposal_prepare(self):
        anchors = self.generate_anchors()
        proposals = np.zeros([self.anchor_size * self.convmap_width * self.convmap_height, 4])
        featureReduceHalf = featureReduce / 2
        for i in range(self.convmap_height):
            h = i * featureReduce + featureReduceHalf
            for j in range(self.convmap_width):
                w = j * featureReduce + featureReduceHalf
                for k in range(self.anchor_size):
                    index = i * self.convmap_width * self.anchor_size + j * self.anchor_size + k
                    anchor = anchors[k, :]
                    proposals[index, :] = anchor + np.array([w, h, w, h])
        proposals[:, 2] -= proposals[:, 0]
        proposals[:, 3] -= proposals[:, 1]
        self.proposals = proposals

    def generate_anchors(self):
        anchors = np.zeros([self.anchor_size, 4])
        for i in range(self.anchor_size):
            anchor_width = self.wandh[i][0]
            anchor_height = self.wandh[i][1]
            anchors[i, :] = np.array(
                [-0.5 * anchor_width, -0.5 * anchor_height, 0.5 * anchor_width, 0.5 * anchor_height])
        return anchors
