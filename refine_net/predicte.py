import sys
import pickle
# sys.path
import os
import glob
from optparse import OptionParser
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from logger.log import debug_logger
from logger.plot import history_ploter
from loss.dice_loss import dice_coeff
from loss.focal_loss import FocalLoss

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torch.nn as nn
from torch import optim
from dataset import predicteCropDataset
from model import UNet, NestedUNet, SCSE_UNet, SCSENestedUNet, DilatedUNet
from utils import  batch, low_bight_filter
from utils import masks_2_labels_maximum, masks_2_labels_coordinate_weight, masks_2_labels_thread, masks_2_labels_max_tread
torch.set_default_tensor_type('torch.cuda.FloatTensor')


# def train_net(net,
#               epochs=5,
#               batch_size=10,
#               lr=0.1,
#               save_cp=True,
#               gpu=False,):
#
#     print('''
#     Starting training:
#         Epochs: {}
#         Batch size: {}
#         Learning rate: {}
#         Training size: {}
#         Validation size: {}
#         Checkpoints: {}
#         CUDA: {}
#     '''.format(epochs, batch_size, lr, len(train_dataset),
#                len(val_dataset), str(save_cp), str(gpu)))
#
#
#     optimizer = optim.SGD(net.parameters(),
#                           lr=lr,
#                           momentum=0.9,
#                           weight_decay=1e-4)
#
#     criterion = nn.BCELoss().cuda()
#     # criterion = nn.BCEWithLogitsLoss()
#     # criterion = nn.CrossEntropyLoss(size_average=True).cuda()
#     # criterion = nn.NLLLoss()
#     # criterion = FocalLoss()
#     landmarks_loss = nn.MSELoss()
#     best_metrics = 10
#     loss_history = []
#     early_stop_count = 0
#     e_loss_best = 100
#     best_dice = 0.0001
#     for epoch in range(epochs):
#         logger.info(f'Epoch: {epoch}')
#         logger.info(f'Learning rate: {lr}')
#         print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
#         net.train()
#         train_losses = []
#
#         with tqdm(train_loader) as _tqdm:
#             for batched in _tqdm:
#                 img_region, label_region, masks_region, begin_crop = batched
#                 # img_region = img_region.type(torch.cuda.FloatTensor)
#                 # masks_region = masks_region.type(torch.cuda.FloatTensor)
#                 # if gpu:
#                 #     img_region, masks_region = img_region.to(device), masks_region.to(device)
#                 optimizer.zero_grad()
#
#                 masks_pred = model(img_region)
#                 # masks_pred = masks_pred.type(torch.cuda.LongTensor)
#                 # masks_pred = low_bight_filter(masks_pred)
#                 # masks_region_one_chan = torch.zeros((10, 250, 250)).type(torch.cuda.FloatTensor)
#                 # for i in range(10):
#                 #     for j in range(19):
#                 #         masks_region_one_chan[i] = masks_region[i][j] * (j + 1) + masks_region_one_chan[i]
#                 # masks_region_one_chan = masks_region_one_chan.type(torch.cuda.LongTensor)
#
#                 loss = criterion(masks_pred, masks_region)
#
#                 _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}'))
#                 train_losses.append(loss.item())
#
#                 loss.backward()
#                 optimizer.step()
#
#
#         # scheduler.step()
#         train_loss = np.mean(train_losses)
#         # train_iou = np.mean(train_ious)
#         logger.info(f'train loss: {train_loss}')
#         # logger.info(f'train iou: {train_iou}')
#
#         valid_losses = []
#         valid_ious = []
#         model.eval()
#         with torch.no_grad():
#             with tqdm(val_loader) as _tqdm:
#                 for batched in _tqdm:
#                     img_region, label_region, masks_region, begin_crop = batched
#
#                     masks_pred = model(img_region)
#
#                     loss = criterion(masks_pred, masks_region)
#
#
#                     _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}'))
#                     valid_losses.append(loss.item())
#         #
#         # # val_dice = eval_net(net, val_loader, gpu)
#         # model.train()
#         valid_loss = np.mean(valid_losses)
#         logger.info(f'valid seg loss: {valid_loss}')
#
#         loss_history.append([train_loss, 0])
#         # loss_history.append([train_loss, valid_loss])
#         history_ploter(loss_history, log_dir.joinpath('loss.png'))
#
#         torch.save(model.state_dict(), dir_checkpoint+'model_tmp.pth')
#         if best_metrics > train_loss:
#
#             best_metrics = train_loss
#             logger.info('Best Model! Early stop recount.')
#             torch.save(model.state_dict(), dir_checkpoint+'Bestmodel_'+str(epoch)+'.pth')
#             early_stop_count = 0
#         else:
#             early_stop_count = early_stop_count + 1
#         logger.info(f'Early stop count: {early_stop_count}')
#         history_dict = {'loss': loss_history,
#                         'best_metrics': best_metrics}
#         with open(log_dir.joinpath('history.pkl'), 'wb') as f:
#             pickle.dump(history_dict, f)
#
#         if early_stop_count == 1000:
#             print('early stop')
#             break

def predicte_region_masks(net, img_regions):
    net.eval()
    with torch.no_grad():
        masks_regions = net(img_regions)
    return masks_regions

def merge_patches(patches, begin_crop_labels, width_full):
    num, channel, width_crop, _ = patches.shape
    margin = width_crop
    width_big = width_full + 2 * margin

    merge_big = torch.zeros(channel, width_big, width_big)
    begin_crop_margin = begin_crop_labels + margin
    end_crop_margin = begin_crop_margin + width_crop
    for n in range(num):
        merge_big_tmp = torch.zeros(channel, width_big, width_big)
        patch = patches[n]

        beg_c = np.array(begin_crop_margin[n]).astype(int)
        end_c = np.array(end_crop_margin[n]).astype(int)
        merge_big_tmp[0:channel, beg_c[1]:end_c[1], beg_c[0]:end_c[0]] = patch
        fold = (merge_big_tmp*merge_big != 0).type(torch.cuda.FloatTensor)
        unfold = (merge_big_tmp*merge_big == 0).type(torch.cuda.FloatTensor)
        # merge_big = merge_big_tmp*(fold/(2)+unfold) + merge_big * (fold/(2)+unfold)
        merge_big = merge_big_tmp + merge_big
    merge_big = merge_big[:, margin:margin+width_full, margin:margin+width_full]
    return merge_big


def merge_tmp_full(img_full, img_full_tmp):

    fold = (img_full * img_full_tmp != 0).type(torch.cuda.FloatTensor)
    unfold = (img_full * img_full_tmp == 0).type(torch.cuda.FloatTensor)
    merge_big = img_full_tmp*(fold/(2)+unfold) + img_full * (fold/(2)+unfold)
    return merge_big

def remove_shadow(masks):
    for i in range(masks.shape[0]-1):
        # masks[i] = masks[i]/masks[i].max()
        masks_sum = masks[0:-1].sum(dim=0)
        mask_tmp = 2*masks[i]-masks_sum
        masks[i] = (mask_tmp)*(mask_tmp>0).float()
    return masks
def normalize(masks):
    for i in range(masks.shape[0]-1):
        masks[i] = masks[i]/(masks[i].max())
        # print(masks[i].max())
        # masks_sum = masks[0:-1].sum(dim=0)
        # mask_tmp = 2*masks[i]-masks_sum
        # masks[i] = (mask_tmp)*(mask_tmp>0).float()
    # masks[-1] = masks[0:-1].sum(dim=0)
    # masks[-1] = masks[-1] - (masks[-1]>1).type(torch.cuda.FloatTensor)*(masks[-1]-1)
    # masks[-1] = 1 - masks[-1]

    masks[-1] = masks[0:-1].sum(dim=0)
    masks[-1] = masks[-1] - (masks[-1]>1).type(torch.cuda.FloatTensor)*(masks[-1]-1)
    masks[-1] = 1 - masks[-1]
    return masks

def predicte_labels_full(data_indx, data_loader, model, times):
    # data_indx = 7
    img_list = list([])
    for time in range(times):
        img_regions, label_regions, begin_crop_labels, img, labels, masks = data_loader.dataset[data_indx]
        masks_region = predicte_region_masks(net=model, img_regions=img_regions)
        # mp_maxmean = masks_region.max(dim=-1)[0].max(dim=-1)[0].mean(dim=0)
        mp_maxmean = masks_region.max(dim=-1)[0].max(dim=-1)[0][:,0:-1].mean(dim=1)

        # for p in range(len(mp_maxmean)):
        #     masks_region[p] = masks_region[p].squeeze() * (masks_region[p] > mp_maxmean[p]).type(torch.cuda.FloatTensor)
        #     for l in range(19):
        #         m = masks_region[p, l].max()
        #         if not m == 0:
        #             masks_region[p, l] = masks_region[p, l] / m
        #
        # masks_region[:, -1] = torch.ones(masks_region[:, -1].shape) - masks_region[:, -1]

        width_full = img.shape[0]
        if time == 0:
            masks_full = merge_patches(masks_region, begin_crop_labels, width_full)
            img_full = merge_patches(img_regions, begin_crop_labels, width_full)
            img_list.append(img_full)
        else:
            masks_full1 = merge_patches(masks_region, begin_crop_labels, width_full)
            img_full1 = merge_patches(img_regions, begin_crop_labels, width_full)
            img_list.append(img_full)
            # masks_full = merge_tmp_full(masks_full, masks_full1)
            # img_full = merge_tmp_full(img_full, img_full1)
            masks_full = masks_full + masks_full1
            img_full = img_full + img_full1
    masks_full = masks_full / times
    img_full = img_full / times
    # masks_full = remove_shadow(masks_full)
    masks_full = normalize(masks_full)
    # masks_full_np = np.array(masks_full)
    # labels_full = masks_2_labels_maximum(np.expand_dims(masks_full_np, axis=0))
    labels_full = masks_2_labels_maximum(masks_full.unsqueeze(dim=0))
    if ((((labels_full-labels)**2).squeeze().sum(dim=1))**0.5).max()>30:
        print('mmm'+str(data_indx))
    # labels_full = masks_2_labels_thread(masks_full.unsqueeze(dim=0))
    return labels_full, img_full, masks_full, img, labels, masks


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5000, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load', default=13, help='load file model')
    # parser.add_option('-c', '--load', dest='load', default=False, help='load file model')
    # parser.add_option('-s', '--scale', dest='scale', type='float', default=0.15, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dir_checkpoint = 'data/legacy/dental/refinenet_model/'
    dir_checkpoint_folders = sorted(glob.glob(dir_checkpoint+'*'))
    folders_num = [int(os.path.split(dir)[-1]) for dir in dir_checkpoint_folders][-1]+1
    if args.load:
        folders_num = int(args.load)
    dir_checkpoint = dir_checkpoint+'%03d'%(folders_num)+'/'
    dir_checkpoint_ = Path(dir_checkpoint)
    dir_checkpoint_.mkdir(exist_ok=True, parents=True)

    train_data_path = 'data/legacy/dental_data/scale0.5/TrainData/'
    test_data_path = 'data/legacy/dental_data/scale0.5/TestData/'
    val_data_path = 'data/legacy/dental_data/scale0.5/ValData/'

    # #preload labels
    coarse_labels = np.load('data/legacy/dental/refine_net/preloadUnet.npy')
    coarse_labels = torch.from_numpy(coarse_labels).type(torch.cuda.FloatTensor)
    coarse_train_labels = coarse_labels[0:150]
    coarse_test_labels = coarse_labels[150:300]
    coarse_val_labels = coarse_labels[300:400]

    # train_dataset = predicteCropDataset(data_path=train_data_path, crop_size=250, train_landmark='all', random_crop=True)
    # train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    #
    val_dataset = predicteCropDataset(data_path=val_data_path,
                                      crop_size=125,
                                      train_landmark='all',
                                      random_crop_range=1.8,
                                      preload_labels='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)

    test_dataset = predicteCropDataset(data_path=test_data_path,
                                       crop_size=200,
                                       train_landmark='all',
                                       random_crop_range=0,
                                       preload_labels=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)

    log_dir = Path(dir_checkpoint+'logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    logger = debug_logger(log_dir)
    logger.info(f'Device: {device}')
    logger.info(f'Max Epoch: {args.epochs}')




    model = UNet(n_channels=1, n_classes=20)
    # model = NestedUNet(n_channels=1, n_classes=20)
    # model = SCSE_UNet(n_channels=1, n_classes=20)
    # model = SCSENestedUNet(n_channels=1, n_classes=20)
    # model = DilatedUNet(in_channels=1, classes=20)


    if args.load:
        load_dirs = sorted(glob.glob(dir_checkpoint+'Bestmodel_*'))
        num = []
        for load_dir in load_dirs:
            model_nums = int(load_dir.split('Bestmodel_')[-1].split('.pth')[0])
            num.append(model_nums)
            # print(model_nums)
        last_best_num = np.array(num).max()
        # last_best_num = sorted(np.array(num))[-2]
        load_dir = dir_checkpoint+'Bestmodel_'+str(last_best_num)+'.pth'
        model.load_state_dict(torch.load(load_dir))
        print('Model loaded from {}'.format(load_dir))

    if args.gpu:
        model.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory


    print('predicting')
    data_loader = test_loader
    data_num = len(data_loader.dataset)
    labels_true_all = torch.cuda.FloatTensor(data_num, 19, 2)
    labels_pred_all = torch.FloatTensor(data_num, 19, 2)
    times = 1
    csv_dir = 'data/legacy/dental/refine_net/pred_csv/'
    for data_index in range(data_num):
        print(str(data_index))
        labels_full, _, _, img, labels, masks = predicte_labels_full(data_index, data_loader, model, times)
        # print(labels_full)
        labels_full = labels_full*2 + torch.cuda.FloatTensor((0, 465))
        labels_true_all[data_index] = labels*2 + torch.cuda.FloatTensor((0, 465))
        labels_pred_all[data_index] = labels_full
        # np.savetxt(csv_dir+'%03d.csv'%(data_index+1), np.array(labels_full).reshape((19, 2)), fmt="%d,%d", delimiter="\n")
    print('evaluate')
    # evaluate
    labels_pred_all = np.array(labels_pred_all)
    labels_true_all = np.array(labels_true_all)
    SDR = np.zeros((20, 4))
    for j in range(19):
        # a = (2 * mse_loss(all_pred_label, all_label)) ** 0.5 / 10
        a = ((((labels_pred_all[:, j, :] - labels_true_all[:, j, :]) ** 2).sum(axis=1)) ** 0.5) / 10
        sdr = [2, 2.5, 3, 4]
        for i in range(4):
            s = (a < sdr[i]).sum() / len(labels_pred_all)
            SDR[j, i] = round(s * 1000)/10
            print(s * 100)
    SDR[19] = SDR[0:19].mean(axis=0)
    import csv
    csvFile = open('csvFile.csv', 'w', newline='')  # 设置newline，否则两行之间会空一行
    writer = csv.writer(csvFile)
    for i in range(20):
        writer.writerow(SDR[i])  # 加一个[]变成array
    csvFile.close()

