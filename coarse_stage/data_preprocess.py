import os
import glob
from mxnet import image
import numpy as np
import cv2

# data_path = "D:\dental\Cephalometry"
data_path = 'data/legacy/dental/Cephalometry_crop'
traindata_path = os.path.join(data_path, 'RawImage/TrainData')
testdata_path = os.path.join(data_path, 'RawImage/TestData')
valdata_path = os.path.join(data_path, 'RawImage/ValData')
trainlist_path = os.path.join(data_path, 'train.lst')
testlist_path = os.path.join(data_path, 'test.lst')
vallist_path = os.path.join(data_path, 'val.lst')
label_path = os.path.join(data_path, 'AnnotationsByMD/400_ave')

def get_data_list(dir):
    data_list = sorted(glob.glob(dir+'/*'))
    return data_list

def crop_save(img_dir):
    img = image.imread(img_dir)
    img_crop = image.fixed_crop(img, 0, 465, 1935, 1935)[:,:,0]
    img_np = img_crop.asnumpy()
    img_dir.split(data_path)
    # img_save_path = save_path
    cv2.imwrite(img_dir, img_np)

def preprocess_img(data_dir):
    data_list = get_data_list(data_dir)
    for img_dir in data_list:
        crop_save(img_dir)

def extract_labels_from_txt(path):
    with open(path, "r") as f:
        # only first 19 are actual coords in dataset label files
        raw = f.readlines()
        coords_raw = raw[:19]
        coords_raw = [tuple([(float(s)) for s in t.split(",")]) for t in coords_raw]
        add_pra = raw[19:]
        add_pra = [s[0] for s in add_pra]
        return coords_raw, add_pra

def preprocess_label(label_path):
    label_files = sorted(glob.glob(label_path+'/*'))
    for file in label_files:
        label_raw = extract_labels_from_txt(file)[0]
        label_np = np.array(label_raw) - [0, 465]
        c = label_np.tolist()
        c_t = [tuple(c_l) for c_l in c]
        c_ts = [str(m).split('(')[1].split(')')[0] for m in c_t]
        c_all = c_ts
        with open(file, "w") as f:
            for line in c_all:
                f.writelines(line + '\n')


print('ok')
preprocess_img(trainlist_path)
preprocess_img(testlist_path)
preprocess_img(vallist_path)
preprocess_label(label_path)
