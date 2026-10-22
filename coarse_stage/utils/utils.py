import random
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

def masks_2_labels_maximum(masks):
    batch_num = masks.shape[0]
    channel_num = masks.shape[1]-1
    labels = np.zeros((batch_num, channel_num, 2))
    for i in range(batch_num):
        for j in range(channel_num):
            max_indx = np.array(np.where(masks[i][j]==masks[i][j].max()))
            labels[i,j,1] = max_indx[0][0]
            labels[i,j,0] = max_indx[1][0]
    return labels

def masks_2_labels_coordinate_weight(masks):
    if masks.dtype == torch.float32:
        masks = masks.type(torch.FloatTensor).numpy()
    batch_num = masks.shape[0]
    channel_num = masks.shape[1]-1
    masks_width = masks.shape[2]
    labels = np.zeros((batch_num, channel_num, 2))
    m = np.array(range(1, masks_width+1)).reshape(1, masks_width)
    m_x = np.repeat(m, masks_width, axis=0)
    # m_x = torch.from_numpy(m_x).type(torch.FloatTensor)
    m_y = np.repeat(m.transpose((1,0)), masks_width, axis=1)
    # m_y = torch.from_numpy(m_y).type(torch.FloatTensor)

    for i in range(batch_num):
        for j in range(channel_num):
            mask_thread_ = masks[i][j] > 0.5
            mx_masks = (m_x*masks[i][j])
            tm_x = mask_thread_ * mx_masks
            labels[i,j,1] = tm_x.sum()/(masks_width**2 - np.where(tm_x==0)[0].shape[0])
            my_masks = (m_y*masks[i][j])
            tm_y = mask_thread_ * my_masks
            labels[i,j,0] = tm_y.sum()/(masks_width**2 - np.where(tm_y==0)[0].shape[0])
    return labels

def masks_2_labels_thread(masks):
    thread = 0.5
    # masks = (masks > 0.3)
    batch_num = masks.shape[0]
    channel_num = masks.shape[1]-1
    masks_width = masks.shape[2]
    labels = np.zeros((batch_num, channel_num, 2))
    for i in range(batch_num):
        for j in range(channel_num):
            mask = masks[i][j]
            # mask = mask/mask.max()
            mask = mask > thread
            mask_labels = np.where(mask == mask.max())
            labels[i,j,0] = mask_labels[1].mean()
            labels[i,j,1] = mask_labels[0].mean()
    return labels

def masks_2_labels_normal_thread(masks):
    thread = 0.5
    # masks = (masks > 0.3)
    batch_num = masks.shape[0]
    channel_num = masks.shape[1]-1
    masks_width = masks.shape[2]
    labels = np.zeros((batch_num, channel_num, 2))
    for i in range(batch_num):
        for j in range(channel_num):
            mask = masks[i][j]
            mask = mask/mask.max()
            mask = mask > thread
            mask_labels = np.where(mask == mask.max())
            labels[i,j,0] = mask_labels[1].mean()
            labels[i,j,1] = mask_labels[0].mean()
    return labels

def masks_2_labels_normal_thread2(masks):
    thread = 0.1
    # masks = (masks > 0.3)
    batch_num = masks.shape[0]
    channel_num = masks.shape[1]-1
    masks_width = masks.shape[2]
    labels = np.zeros((batch_num, channel_num, 2))
    for i in range(batch_num):
        for j in range(channel_num):
            mask = masks[i][j]
            mask = mask/mask.max()
            mask = mask > thread
            mask_labels = np.where(mask == mask.max())
            labels[i,j,0] = mask_labels[1].mean()
            labels[i,j,1] = mask_labels[0].mean()
    return labels

def masks_2_labels_max_tread(masks):
    max_label = masks_2_labels_maximum(masks)
    nor_thread_laebl = masks_2_labels_normal_thread(masks)
    # nor_thread_laebl2 = masks_2_labels_normal_thread2(masks)
    mean_label = (max_label + nor_thread_laebl)/2
    return mean_label