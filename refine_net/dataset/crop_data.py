from functools import partial
import numpy as np

from pathlib import Path

import torch
import random
from torch.utils.data import DataLoader, Dataset

from utils.preprocess import minmax_normalize, meanstd_normalize, padding
from utils import load_all_data_from_npy, get_masks, data_augmentation, region_crop, load_preload_labels


torch.set_default_tensor_type('torch.cuda.FloatTensor')



class CropDataset(Dataset):
    def __init__(self, data_path, crop_size, train_landmark, random_crop_range, mutli_channels):
        self.img_all, self.label_all = load_all_data_from_npy(data_path)
        self.img_size = self.img_all[0].shape[0]
        self.data_aug = data_augmentation()
        self.crop_size = crop_size
        self.train_landmark = train_landmark
        self.random_crop = random_crop_range
        self.mutli_channels = mutli_channels

    def __len__(self):
        return len(self.img_all)

    def __getitem__(self, index):
        if not (self.train_landmark in range(19)):
            # print('train all landmarks')
            self.landmark_num = random.randint(0, 18)
        else:
            self.landmark_num = self.train_landmark
        img = self.img_all[index]
        labels = self.label_all[index]
        labels_one = labels[self.landmark_num]
        if not self.mutli_channels:
            labels = labels_one
        masks = get_masks(labels, self.img_size)
        img_region, label_region, masks_region, begin_crop = region_crop(img, labels_one, masks, self.crop_size, self.random_crop)


        return img_region, label_region, masks_region, begin_crop

class predicteCropDataset(Dataset):
    def __init__(self, data_path, crop_size, train_landmark, random_crop_range, preload_labels=None):
        self.img_all, self.label_all = load_all_data_from_npy(data_path)
        self.img_size = self.img_all[0].shape[0]
        self.data_aug = data_augmentation()
        self.crop_size = crop_size
        self.train_landmark = train_landmark
        self.random_crop = random_crop_range
        # self.preload_dir = "data/legacy/user\Desktop\dental/refine_net\preloaddcUnet.npy"
        self.preload_dir = 'data/legacy/dental/refine_net/preloadunet.npy'
        # self.preload_labels = load_preload_labels(self.preload_dir)
        if preload_labels == 'train':
            self.preload_labels, _, _ = load_preload_labels(self.preload_dir)
        elif preload_labels == 'test':
            _, self.preload_labels, _ = load_preload_labels(self.preload_dir)
        elif preload_labels == 'val':
            _, _, self.preload_labels = load_preload_labels(self.preload_dir)
        else:
            self.preload_labels = np.array(0)

        if  (self.train_landmark in range(19)):
            print('predict ' + str(self.train_landmark)+' landmark')
        elif self.train_landmark == 'all':
            landmark_total_num = self.label_all[0].__len__()
            print('predict all ' + str(landmark_total_num)+' landmarks')

    def __len__(self):
        return len(self.img_all)

    def __getitem__(self, index):

        img = self.img_all[index]
        labels = self.label_all[index]
        masks = get_masks(labels, self.img_size)

        if  (self.train_landmark in range(19)):
            # print('predict '+ str(self.train_landmark)+' landmark')
            labels_one = labels[self.train_landmark]
            img_regions, label_regions, _,  begin_crop_labels = region_crop(img, labels_one, masks, self.crop_size, self.random_crop)
        elif self.train_landmark == 'all':
            landmark_total_num = labels.__len__()
            # print('predict all '+ str(landmark_total_num)+'landmarks')
            img_regions = torch.cuda.FloatTensor(landmark_total_num, 1, self.crop_size, self.crop_size)
            label_regions = torch.cuda.FloatTensor(landmark_total_num, 2)
            begin_crop_labels = torch.cuda.FloatTensor(landmark_total_num, 2)
            if torch.is_tensor(self.preload_labels):
                labels_t = self.preload_labels[index]
            else:
                labels_t = labels
            for i, labels_one in enumerate(labels_t):
                img_region, label_region, _, begin_crop = region_crop(img, labels_one, masks, self.crop_size, self.random_crop)
                img_regions[i] = img_region
                label_regions[i] = label_region
                begin_crop_labels[i] = begin_crop
        else:
            print('landmarks error')

        return img_regions, label_regions, begin_crop_labels, img, labels, masks
