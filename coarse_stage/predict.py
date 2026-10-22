import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import glob
from PIL import Image
#
# from Pytorch_UNet.unet import UNet, NestedUNet, SCSE_UNet, SCSENestedUNet
# from Pytorch_UNet.utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
# from Pytorch_UNet.utils import masks_2_labels_maximum, masks_2_labels_coordinate_weight, masks_2_labels_thread, masks_2_labels_normal_thread, masks_2_labels_normal_thread2, masks_2_labels_max_tread
# from Pytorch_UNet.utils import plot_img_and_mask

from unet import UNet, NestedUNet, SCSE_UNet, SCSENestedUNet, DilatedUNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
from utils import masks_2_labels_maximum, masks_2_labels_coordinate_weight, masks_2_labels_thread, masks_2_labels_normal_thread, masks_2_labels_normal_thread2, masks_2_labels_max_tread
from utils import plot_img_and_mask

from torchvision import transforms

def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):

    net.eval()

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

    # left_square, right_square = split_img_into_squares(img)

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
    # return full_mask > out_threshold



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

# if __name__ == "__main__":
# args = get_args()
# in_files = args.input
# dir_img = "D:\dental\Cephalometry_crop\RawImage/TrainData/"# 1-150
dir_img = 'data/legacy/dental/Cephalometry_crop/RawImage/TestData/' # 151-300
# dir_img = "D:\dental\Cephalometry_crop\RawImage/valData/" # 301-400


# dir_img = "D:\dental_hospital/"

class args:
    def __init__(self):
        # self.model = "D:\dental/unetpp_focal20_softmax/CP441.pth"
        # self.model = "D:\dental/dcunet_model_40\CP551.pth"
        self.model = 'data/legacy/dental/unet_model_2loss_40/CP468.pth'
        # self.model = "D:\dental/unet_model\CP435.pth"
        # self.model = "D:\dental/unet_model\CP226.pth"
        # self.model = "D:\dental/unetpp_40-10_softmax\min_loss.pth"

        self.input = sorted(glob.glob(dir_img+'*'))
        self.output = sorted(glob.glob(dir_img+'*'))
        self.cpu = False
        self.viz = False
        self.no_save = False
        self.no_crf = False
        self.mask_threshold = False

        self.scale = float(0.15)

args = args()

in_files = sorted(glob.glob(dir_img+'*'))
# args.input = in_files
# out_files = get_output_filenames(args)

net = UNet(n_channels=1, n_classes=20)
# net = NestedUNet(n_channels=1, n_classes=20)
# net = SCSENestedUNet(n_channels=1, n_classes=20)
# net = DilatedUNet(in_channels=1, classes=20)

print("Loading model {}".format(args.model))

if not args.cpu:
    print("Using CUDA version of the net, prepare your GPU !")
    net.cuda()
    net.load_state_dict(torch.load(args.model))
else:
    net.cpu()
    net.load_state_dict(torch.load(args.model, map_location='cpu'))
    print("Using CPU version of the net, this may be very slow")

print("Model loaded !")

pred_label_all = np.zeros((len(in_files), 19, 2))
true_label_all = np.zeros((len(in_files), 19, 2))
csv_dir = 'data/legacy/dental/Pytorch_UNet/pred_csv/'
for i, fn in enumerate(in_files):
    # print("\nPredicting image {} ...".format(fn))
    img = Image.open(fn)
    img = img.convert('L')

    # img = img.crop((266, 0, 2570, 2304))
    img = img.resize((1935, 1935))

    if img.size[0] < img.size[1]:
        print("Error: image height larger than the width")

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       use_dense_crf= not args.no_crf,
                       use_gpu=not args.cpu)
    # pred_label = masks_2_labels_maximum(mask.reshape((1,).__add__(mask.shape)))
    pred_label = masks_2_labels_normal_thread(mask.reshape((1,).__add__(mask.shape)))
    # pred_label = masks_2_labels_max_tread(mask.reshape((1,).__add__(mask.shape)))

    # evaluate
    pred_label_all[i] = pred_label/args.scale
    true_label_all[i] = get_true_label(fn)
    if (pred_label_all[i] - true_label_all[i]).max()>100:
        print(i)
    # np.savetxt(csv_dir+'%03d.txt'%(i+1), np.array(pred_label_all[i]/2).reshape((19, 2)), fmt="%.2f,%.2f", delimiter="\n")
    # np.savetxt(csv_dir+'%03d.txt'%(i+151), np.array(pred_label_all[i]/2).reshape((19, 2)), fmt="%.2f,%.2f", delimiter="\n")
    # np.savetxt(csv_dir+'%03d.txt'%(i+301), np.array(pred_label_all[i]/2).reshape((19, 2)), fmt="%.2f,%.2f", delimiter="\n")

    # from matplotlib import pyplot as plt
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    # ax1.imshow(img, cmap="gray")
    # pred_label = pred_label.reshape((19, 2))*5
    # for c in pred_label:
    #     # add patches to original image
    #     # could also just plt.scatter() but less control then
    #     ax1.add_patch(plt.Circle(c, 5, color='r'))
    #
    # true_label = get_true_label(fn)
    # true_label = true_label.reshape((19, 2))
    # for c in true_label:
    #     # add patches to original image
    #     # could also just plt.scatter() but less control then
    #     ax1.add_patch(plt.Circle(c, 5, color='b'))
    # plt.show()

    # if args.viz:
    #     print("Visualizing results for image {}, close to continue ...".format(fn))
    #     plot_img_and_mask(img, mask)

    # if not args.no_save:
    #     out_fn = out_files[i]
    #     result = mask_to_image(mask)
    #     result.save(out_files[i])

        # print("Mask saved to {}".format(out_files[i]))


# evaluate
SDR = np.zeros((20, 5))
for j in range(19):
    # a = (2 * mse_loss(all_pred_label, all_label)) ** 0.5 / 10
    a = ((((pred_label_all[:, j, :] - true_label_all[:, j, :]) ** 2).sum(axis=1)) ** 0.5) / 10
    sdr = [2, 2.5, 3, 4, 10]
    for i in range(5):
        s = (a < sdr[i]).sum() / len(pred_label_all)
        SDR[j, i] = round(s * 1000)/10
        print(s * 100)
SDR[19] = SDR[0:19].mean(axis=0)
import csv
csvFile = open('csvFile.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
writer = csv.writer(csvFile)
for i in range(20):
    writer.writerow(SDR[i])  # 加一个[]变成array
csvFile.close()

def predict_mask_label(img):

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       use_dense_crf= not args.no_crf,
                       use_gpu=not args.cpu)
    pred_label = masks_2_labels_max_tread(mask.reshape((1,).__add__(mask.shape)))
    pred_label = pred_label/args.scale
    pred_label = np.array(pred_label)
    # plot_img_and_mask(img, mask)
    return pred_label
