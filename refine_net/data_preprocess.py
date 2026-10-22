import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import cv2
import glob
from PIL import Image
import random

from model import UNet, NestedUNet, SCSE_UNet, SCSENestedUNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
from utils import masks_2_labels_maximum, masks_2_labels_coordinate_weight, masks_2_labels_thread, masks_2_labels_normal_thread, masks_2_labels_normal_thread2, masks_2_labels_max_tread
from utils import plot_img_and_mask


def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):

    net.eval()

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

    img = hwc_to_chw(img)

    img = torch.from_numpy(img).unsqueeze(0)
    img = img.type(torch.FloatTensor)

    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        output_masks = net(img)

        output_masks = output_masks.squeeze(0)

        output_masks = output_masks.cpu().numpy()
    full_mask = output_masks

    # if use_dense_crf:
    #     full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    return full_mask

def get_true_label(imgpath):
    labelpath = 'data/legacy/dental/Cephalometry_crop/AnnotationsByMD/400_ave/'
    labels = np.zeros((1, 19, 2))
    cls = imgpath.split('\\')
    index_num = cls[-1].split('.bmp')[0]
    with open(os.path.join(labelpath, '%03d.txt'% (int(index_num)))) as f:
        coords_raw = f.readlines()[:19]
        p = []
        for t in coords_raw:
            for f in t.split(','):
                p.append(float(f))
        p = np.array(p).reshape(1,len(p))
        labels = p.reshape((1,19,2))

    return labels

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def get_true_label(imgpath):
    labelpath = 'data/legacy/dental/Cephalometry_crop/AnnotationsByMD/400_ave/'
    labels = np.zeros((1, 19, 2))
    cls = imgpath.split('\\')
    index_num = cls[-1].split('.bmp')[0]
    with open(os.path.join(labelpath, '%03d.txt'% (int(index_num)))) as f:
        coords_raw = f.readlines()[:19]
        p = []
        for t in coords_raw:
            for f in t.split(','):
                p.append(float(f))
        p = np.array(p).reshape(1,len(p))
        labels = p.reshape((1,19,2))

    return labels

def img_crop(img, begin_crop, size):
    return img[begin_crop[0]: begin_crop[0]+size, begin_crop[1]:begin_crop[1]+size]

def img_centercrop(img, center_label, size):
    img = np.array(img)

    x0 = int(center_label[0]-size/2)
    x1 = int(center_label[0]+size/2)
    y0 = int(center_label[1]-size/2)
    y1 = int(center_label[1]+size/2)
    return img[x0:x1, y0:y1]


def label_centercrop(true_label, pred_label, size):
    return true_label-pred_label+size/2

def local_crop(img, true_labels, pred_labels, size=160):
    true_labels = true_labels.reshape((19,2))
    pred_labels = pred_labels.reshape((19,2))

    pred_label_rand = np.zeros((2))
    img_crops = np.zeros((19, size, size))
    label_crops = np.zeros((19,2))
    img_location = np.zeros((19,2))
    for i, pred_label in enumerate(pred_labels):
        if args.rand:
            random_x0 = random.randint(-50, 50)
            random_y0 = random.randint(-50, 50)
        else:
            random_x0 = 0
            random_y0 = 0
        pred_label_rand[0] = pred_label[0] + random_x0
        pred_label_rand[1] = pred_label[1] + random_y0
        img_crops[i] = img_centercrop(img, pred_label_rand, size)
        label_crops[i] = label_centercrop(true_labels[i], pred_label_rand, size)
        img_location[i] = pred_label_rand - size/2

    return img_crops, label_crops, img_location

dir_img = 'data/legacy/dental/Cephalometry_crop/RawImage/TrainData/'
# dir_img = "D:\dental\Cephalometry_crop\RawImage/TestData/"
# dir_img = "D:\dental\Cephalometry_crop\RawImage/ValData/"
class args:
    def __init__(self):
        # self.net = UNet(n_channels=1, n_classes=20)
        self.net = NestedUNet(n_channels=1, n_classes=20)
        # self.net = SCSENestedUNet(n_channels=1, n_classes=20)
        self.model = 'data/legacy/dental/unetpp_focal20_softmax/CP441.pth'
        self.input = sorted(glob.glob(dir_img+'*'))
        self.output = sorted(glob.glob(dir_img+'*'))
        self.cpu = False
        self.viz = False
        self.no_save = False
        self.no_crf = False
        self.mask_threshold = False
        self.crop_size = 600
        self.scale = 0.15
        self.rand = True
args = args()
in_files = sorted(glob.glob(dir_img+'*'))

print("Loading model {}".format(args.model))

if not args.cpu:
    print("Using CUDA version of the net, prepare your GPU !")
    args.net.cuda()
    args.net.load_state_dict(torch.load(args.model))
else:
    args.net.cpu()
    args.net.load_state_dict(torch.load(args.model, map_location='cpu'))
    print("Using CPU version of the net, this may be very slow")

print("Model loaded !")
data_num = len(in_files)
print("data num is "+str(data_num))

img_crops_all = np.zeros((data_num, 19, args.crop_size, args.crop_size))
label_crops_all = np.zeros((data_num, 19, 2))
img_location_all = np.zeros((data_num, 19, 2))
for i, fn in enumerate(in_files):
    print("\nPredicting image {} ...".format(fn))
    img = Image.open(fn)
    img = img.convert('L')

    # img = img.crop((266, 0, 2570, 2304))
    img = img.resize((1935, 1935))

    if img.size[0] < img.size[1]:
        print("Error: image height larger than the width")

    masks = predict_img(net=args.net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       use_dense_crf= not args.no_crf,
                       use_gpu=not args.cpu)

    pred_labels = masks_2_labels_maximum(masks.reshape((1,).__add__(masks.shape)))/args.scale

    labels = get_true_label(fn)

    img_crops, label_crops, img_location = local_crop(img, labels, pred_labels, size=args.crop_size)
    img_crops_all[i] = img_crops
    label_crops_all[i] = label_crops
    img_location_all[i] = img_location
# dir_cropdata = "D:\dental_data/refinenet_data\crop80dataset/train_crop80_data/"
# dir_cropdata = "D:\dental_data/refinenet_data\crop80dataset/test_crop80_data/"
# dir_cropdata = "D:\dental_data/refinenet_data\crop80dataset/val_crop80_data/"

dir_cropdata = 'data/legacy/dental_data/refinenet_data/crop160dataset/train_crop_data/'
# dir_cropdata = "D:\dental_data/refinenet_data\crop160dataset/test_crop_data/"
# dir_cropdata = "D:\dental_data/refinenet_data\crop160dataset/val_crop_data/"
np.save(dir_cropdata+'img_crops', img_crops_all)
np.save(dir_cropdata+'label_crops', label_crops_all)
np.save(dir_cropdata+'img_location', img_location_all)
print('data preprocess finish')

