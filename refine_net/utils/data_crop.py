import os

import numpy as np
import torch
from PIL import Image
import cv2
import math
import glob
import random
import collections
from .utils import resize_and_crop, get_square, normalize, hwc_to_chw
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def label_2_mask(label_scale, img_width_scaled, normal_dist):
    label_scale = label_scale.type(torch.cuda.IntTensor)
    img_width_scaled = int(img_width_scaled)
    normal_dist_width = int(normal_dist.shape[0])
    normal_dist_half_width = int(normal_dist.shape[0]/2)
    extend_width = int(img_width_scaled + normal_dist_width)
    mask_extend = torch.zeros(extend_width, extend_width)
    nor_x_begin = int(label_scale[1])
    nor_x_end = int(label_scale[1] + normal_dist_width)
    nor_y_begin = int(label_scale[0])
    nor_y_end = int(label_scale[0] + normal_dist_width)
    mask_extend[nor_x_begin:nor_x_end, nor_y_begin:nor_y_end] = normal_dist
    mask_begin = int(normal_dist_half_width)
    mask_end = int(mask_begin+img_width_scaled)
    mask = mask_extend[mask_begin:mask_end, mask_begin:mask_end]
    return mask

def normal_distribution(size=400):
    res = 6/size
    x = torch.arange(-3, 3, res)
    y = torch.arange(-3, 3, res)


    x, y = torch.meshgrid((x, y))
    z = (1 / (2 * math.pi * 1 ** 2)) * torch.exp(-(x ** 2 + y ** 2) / 2 * 1 ** 2)
    z = ((z / z.max()))
    # z = torch.from_numpy(z).type(torch.cuda.FloatTensor)
    return z

def gennerate_masks(labels, size):
    """From a list of labels, Generate a set of high light masks"""
    labels = labels.reshape((-1, 2))
    masks_channels = labels.shape[0]
    # labels = labels.type(torch.FloatTensor)
    # labels = labels.to('cuda:0')
    # print('ok')

    # 20th masks
    masks = torch.zeros((masks_channels+1, size, size))
    # masks = masks.to('cuda:0')
    for i, label in enumerate(labels):
        masks[i] = label_2_mask(label, size, normal_dist)
    masks[masks_channels] = torch.ones((size, size)) - masks.sum((0))
    # masks[masks_channels] = masks[masks_channels] * ((masks[masks_channels] >= 0).type(torch.cuda.FloatTensor)) # 令小于0的元素等于零

    # 19masks
    # masks = np.zeros((19, size, size))
    # for i, label in enumerate(labels):
    #     masks[i] = label_2_mask(label, size, normal_dist)
    return masks

def get_masks(labels, size):

    masks = gennerate_masks(labels, size)

    # if data_aug:
    #     masks_and_label = map(x.masks_label_aug, masks_and_label)
    return masks

# # normal_dist = normal_distribution(40)
# expand_deque = torch.FloatTensor(((0,0),(-1,-1),(-1,1),(1,-1),(1,1)))
expand_deque = collections.deque([(0,0),(-1,-1),(-1,1),(1,-1),(1,1)])
# expand_deque = collections.deque([(-1,-1),(-1,1),(1,-1),(1,1)])
normal_dist = normal_distribution(90)
def region_crop(img, label, masks, crop_size, random_crop_range):
    img_width = img.shape[1]
    margin = crop_size
    half_width = int(crop_size * 0.5)
    if not crop_size > 0:
        crop_size = img_width
        half_width = int(crop_size * 0.5)
        label = label*0 + half_width

    big_img_width = img_width+2*margin
    if 1 > random_crop_range > 0:
        random_range = int(crop_size * random_crop_range)
        random_x0 = random.randint(-random_range, random_range)
        random_y0 = random.randint(-random_range, random_range)
    elif 2 >= random_crop_range > 1:
        expand = random_crop_range - 1
        overlap = (1-expand)
        shift = 0.5-overlap
        # random_x0=0
        # random_y0=0
        random_x0, random_y0 = np.array(expand_deque[0])*shift*crop_size
        random_x0, random_y0 = int(random_x0), int(random_y0)
        expand_deque.rotate(-1)
    elif random_crop_range == 0:
        random_x0 = 0
        random_y0 = 0

    cropbegin_x0 = int(label[0] + random_x0 - half_width)
    cropbegin_x0_maigin = cropbegin_x0 + margin
    cropbegin_y0 = int(label[1] + random_y0 - half_width)
    cropbegin_y0_maigin = cropbegin_y0 + margin
    begin_crop = torch.cuda.FloatTensor((cropbegin_x0, cropbegin_y0))
    cropend_x0_maigin = int(cropbegin_x0_maigin + crop_size)
    cropend_y0_maigin = int(cropbegin_y0_maigin + crop_size)
    imgbig = torch.cuda.FloatTensor(big_img_width,big_img_width)
    imgbig[margin:margin+img_width,margin:margin+img_width] = img
    img_crop = imgbig[cropbegin_y0_maigin:cropend_y0_maigin, cropbegin_x0_maigin:cropend_x0_maigin]
    img_crop = img_crop.reshape((1, crop_size, crop_size))
    label_crop_relative = -torch.cuda.FloatTensor([random_x0, random_y0])/half_width
    channels_num = masks.shape[0]
    masksbig = torch.zeros((channels_num, big_img_width, big_img_width))
    masksbig[:,margin:margin+img_width,margin:margin+img_width] = masks
    masks_crop = masksbig[:, cropbegin_y0_maigin:cropend_y0_maigin, cropbegin_x0_maigin:cropend_x0_maigin]

    return img_crop, label_crop_relative, masks_crop, begin_crop