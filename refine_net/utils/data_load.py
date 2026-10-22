#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import cv2
import math
import random
import glob
import torch

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def load_all_data_from_npy(data_path):
    images_all = np.load(glob.glob(data_path+'img_*')[0])
    images_all = torch.from_numpy(images_all).type(torch.cuda.FloatTensor)
    # images_all = torch.load()
    labels_all = np.load(glob.glob(data_path+'label_*')[0])
    labels_all = torch.from_numpy(labels_all).type(torch.cuda.FloatTensor)
    return images_all, labels_all

def load_preload_labels(data_path):
    coarse_labels = np.load(data_path)
    coarse_labels = torch.from_numpy(coarse_labels).type(torch.cuda.FloatTensor)
    coarse_train_labels = coarse_labels[0:150]
    coarse_test_labels = coarse_labels[150:300]
    coarse_val_labels = coarse_labels[300:400]
    return coarse_train_labels, coarse_test_labels, coarse_val_labels



class data_augmentation:
    def __init__(self, ):
        self.img_width = 160
        self.img_width_scaled = int(1 * self.img_width)
        self.height, self.width = self.img_width_scaled, self.img_width_scaled
        self.center = (self.width / 2., self.height / 2.)
        # affrat data_augmentation
        self.affrat = random.uniform(0.8, 1.3)
        self.halfl_w = min(self.width - self.center[0], (self.width - self.center[0]) * self.affrat)
        self.halfl_h = min(self.height - self.center[1], (self.height - self.center[1]) * self.affrat)
        # rotated augmentation
        self.angle = random.uniform(0, 45)
        if random.randint(0, 1):
            self.angle *= -1
        self.rotMat = cv2.getRotationMatrix2D(self.center, self.angle, 1.0)



    def img_aug(self, imgs):
        for i, img in enumerate(imgs):
            img = cv2.resize(img[int(self.center[1] - self.halfl_h): int(self.center[1] + self.halfl_h + 1),
                              int(self.center[0] - self.halfl_w): int(self.center[0] + self.halfl_w + 1)], (self.width, self.height))
            img = cv2.warpAffine(img, self.rotMat, (self.width, self.height))
            img = img.reshape((self.height, self.width, 1))
            imgs[i] = np.array(img).squeeze()
        return imgs

    def masks_aug(self, masks):

        for i in range(masks.shape[0]):
            mask = masks[i]
            mask = cv2.resize(mask[int(self.center[1] - self.halfl_h): int(self.center[1] + self.halfl_h + 1),
                              int(self.center[0] - self.halfl_w): int(self.center[0] + self.halfl_w + 1)], (self.width, self.height))
            if not i == 19:
                mask = cv2.warpAffine(mask, self.rotMat, (self.width, self.height))
            elif i==19:
                mask = cv2.warpAffine(mask, self.rotMat, (self.width, self.height), borderValue=1)

            masks[i] = mask
            return masks

    def labels_aug(self, labels):

        for i in range(labels.__len__()):
            annot = labels
            x = annot[i][0]
            y = annot[i][1]
            x = (x - self.center[0]) / self.halfl_w * (self.width - self.center[0]) + self.center[0]
            y = (y - self.center[1]) / self.halfl_h * (self.height - self.center[1]) + self.center[1]
            if x >= 0 and y >= 0:
                R = self.rotMat[:, : 2]
                W = np.array([self.rotMat[0][2], self.rotMat[1][2]])
                annot[i] = np.dot(R, annot[i]) + W

        labels = annot
        return labels