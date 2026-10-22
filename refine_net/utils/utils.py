import random
import os
import numpy as np
import torch


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.5, final_height=None):

    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    if newW % 2 != 0:
        newW = newW + 1
    newH = int(h * scale)
    if newH % 2 != 0:
        newH = newH + 1

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    # img = img.crop((0, diff // 2, newW, newH - diff // 2))
    img = np.array(img, dtype=np.float16).reshape((newW, newH, 1))
    return img

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_shuffle_data(dataset, shuffle=False):
    dataset = list(dataset)
    if shuffle == True:
        random.shuffle(dataset)
    return dataset


def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs



def low_bight_filter(data):
    data_len = data.shape[0]
    for i in range(data_len):
        if not (i == data_len-1):
            data[i] = data[i]*(data[i] > data[i].mean()).type(torch.cuda.FloatTensor)
        if i == data_len-1:
            data[i] = data[i]
    return data

def max_mean_filter(data):
    data_len = data.shape[0]
    max_mean = data.max(axis=1).max(axis=1)[0:data_len-1].mean()
    for i in range(data_len):
        if not (i == data_len-1):
            data[i] = data[i]*(data[i] > max_mean)
        if i == data_len-1:
            data[i] = data[i]
    return data