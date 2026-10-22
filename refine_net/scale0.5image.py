import os
import glob
import numpy as np
from PIL import Image
from utils import resize_and_crop
import torch

# dir_img = "D:\dental\Cephalometry_crop\RawImage/TrainData/"
# dir_img = "D:\dental\Cephalometry_crop\RawImage/TestData/"
dir_img = 'data/legacy/dental/Cephalometry_crop/RawImage/ValData/'
in_files = sorted(glob.glob(dir_img+'*'))
labelpath = 'data/legacy/dental/Cephalometry_crop/AnnotationsByMD/400_ave/'


def get_true_label(imgpath):

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

data_num = len(in_files)
print("data num is "+str(data_num))
scale = 0.5
example = in_files[0]
example = Image.open(example)
example = example.convert('L')
example_scaled = resize_and_crop(example, scale)
img_crops_all = np.zeros((data_num, example_scaled.shape[0], example_scaled.shape[0]))
label_crops_all = np.zeros((data_num, 19, 2))
# img_location_all = np.zeros((data_num, 19, 2))
for i, fn in enumerate(in_files):
    print("\nPredicting image {} ...".format(fn))
    img = Image.open(fn)
    img = img.convert('L')

    img = img.resize((1935, 1935))
    img_scaled = resize_and_crop(img, scale)
    img_crops_all[i] = img_scaled.squeeze()/255
    label_crops_all[i] = get_true_label(fn)*scale

dir_cropdata = 'data/legacy/dental_data/scale0.5/'
dir_cropdata = dir_cropdata + dir_img.split('data/legacy/dental/Cephalometry_crop/RawImage/')[-1]
# dir_cropdata = "D:\dental_data/refinenet_data\crop160dataset/test_crop_data/"
# dir_cropdata = "D:\dental_data/refinenet_data\crop160dataset/val_crop_data/"

if not os.path.exists(dir_cropdata):
    os.makedirs(dir_cropdata)



np.save(dir_cropdata+'img_scale0.5.npy', img_crops_all)
np.save(dir_cropdata+'label_scale0.5.npy', label_crops_all)
# torch.save(img_crops_all, dir_cropdata+'img_scale0.5.npy')
# torch.save(label_crops_all, dir_cropdata+'label_scale0.5.npy')
# np.save(dir_cropdata+'img_location', img_location_all)
print('data preprocess finish')

# predlabelpath = 'data/legacy/user\Desktop\dental\Pytorch_UNet\pred_csv/'
# labels = np.zeros((400, 19, 2))
# for index_num in range(400):
#     with open(os.path.join(predlabelpath, '%03d.txt' % (int(index_num + 1)))) as f:
#         coords_raw = f.readlines()[:19]
#         p = []
#         for t in coords_raw:
#             for f in t.split(','):
#                 p.append(float(f))
#         p = np.array(p).reshape(1, len(p))
#     labels[index_num] = p.reshape((19, 2))
