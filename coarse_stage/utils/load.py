#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
import cv2
import math
import random

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        im = im.astype('float32')
        yield get_square(im, 0)
        # return get_square(im, 0)

def to_cropped_masks(ids, dir, scale, img_width_scaled, normal_dist):
    """From a list of labels, Generate a set of high light masks"""
    for id in ids:
        label_file = dir + id + '.txt'
        with open(os.path.join(label_file)) as f:
            coords_raw = f.readlines()[:19]
            p = []
            for t in coords_raw:
                for f in t.split(','):
                    p.append(float(f))
            # p = mx.ndarray.array(p).reshape(1,len(p))
            # print(p)
        labels = np.array(p).reshape((19,2))
        # mask_labels = labels[[0,1,2,3,8,9,10,16,17]]*scale # 10 channels
        mask_labels = labels*scale # 20 channels
        mask_labels = mask_labels.astype('int')
        # print('ok')
        masks = np.zeros((20, img_width_scaled, img_width_scaled))
        for i, label in enumerate(mask_labels):
            masks[i] = label_2_mask(label, img_width_scaled, normal_dist)
        masks[19] = np.ones((img_width_scaled, img_width_scaled)) - masks.sum(axis=0)
        # masks[19] = masks[19] * (masks[19] >= 0) # 令小于0的元素等于零
        # masks[19] = (masks.sum(axis=0) == 0).astype('float64')
        # masks = masks.transpose([1, 2, 0])
        # return [masks.astype('float32'), mask_labels]
        yield [masks.astype('float32'), mask_labels]


def normal_distribution(size=400):
    res = 6/size
    x = np.arange(-3, 3, res)
    y = np.arange(-3, 3, res)

    x, y = np.meshgrid(x, y)
    z = (1 / (2 * math.pi * 1 ** 2)) * np.exp(-(x ** 2 + y ** 2) / 2 * 1 ** 2)
    z = ((z / z.max())).astype('float32')
    return z


def label_2_mask(label_scale, img_width_scaled, normal_dist):
    label_scale = label_scale.astype('int')
    img_width_scaled = int(img_width_scaled)
    normal_dist_width = int(normal_dist.shape[0])
    normal_dist_half_width = int(normal_dist.shape[0]/2)
    extend_width = int(img_width_scaled + normal_dist_width)
    mask_extend = np.zeros((extend_width, extend_width))
    nor_x_begin = int(label_scale[1])
    nor_x_end = int(label_scale[1] + normal_dist_width)
    nor_y_begin = int(label_scale[0])
    nor_y_end = int(label_scale[0] + normal_dist_width)
    mask_extend[nor_x_begin:nor_x_end, nor_y_begin:nor_y_end] = normal_dist
    mask_begin = int(normal_dist_half_width)
    mask_end = int(mask_begin+img_width_scaled)
    mask = mask_extend[mask_begin:mask_end, mask_begin:mask_end]
    mask = mask
    return mask
def get_imgs_and_masks(ids, dir_img, dir_label, scale, data_aug=False):
    """Return all the couples (img, mask)"""
    ids = list(ids)
    if not ids:
        raise ValueError(f'No .bmp images found in {dir_img}')
    with Image.open(os.path.join(dir_img, ids[0] + '.bmp')) as sample:
        image_width = sample.height
    img_width_scaled = int(image_width * scale)
    if img_width_scaled % 2 != 0:
        img_width_scaled += 1
    imgs = to_cropped_imgs(ids, dir_img, '.bmp', scale)



    if data_aug:
        x = data_augmentation()
        imgs = map(x.img_aug, imgs)


    img_width_origin = image_width
    normal_dist = normal_distribution(40)
    masks_and_label = to_cropped_masks(ids, dir_label, scale, img_width_scaled, normal_dist)
    if data_aug:
        masks_and_label = map(x.masks_label_aug, masks_and_label)

    # need to transform from HWC to CHW/
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)


    return zip(imgs_normalized, masks_and_label)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)


class data_augmentation:
    def __init__(self):
        self.img_width = 1935
        self.img_width_scaled = int(0.2 * self.img_width)
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
        imgs = cv2.resize(imgs[int(self.center[1] - self.halfl_h): int(self.center[1] + self.halfl_h + 1),
                          int(self.center[0] - self.halfl_w): int(self.center[0] + self.halfl_w + 1)], (self.width, self.height))
        imgs = cv2.warpAffine(imgs, self.rotMat, (self.width, self.height))
        imgs = imgs.reshape((self.height, self.width, 1))
        return imgs

    def masks_label_aug(self, masks_and_label):
        masks = masks_and_label[0]
        labels = masks_and_label[1]
        for i in range(masks.shape[0]):
            mask = masks[i]
            mask = cv2.resize(mask[int(self.center[1] - self.halfl_h): int(self.center[1] + self.halfl_h + 1),
                              int(self.center[0] - self.halfl_w): int(self.center[0] + self.halfl_w + 1)], (self.width, self.height))
            if not i == 19:
                mask = cv2.warpAffine(mask, self.rotMat, (self.width, self.height))
            elif i==19:
                mask = cv2.warpAffine(mask, self.rotMat, (self.width, self.height), borderValue=1)

            masks[i] = mask

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
        masks_and_label = [masks, labels]
        return masks_and_label