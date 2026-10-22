from optparse import OptionParser

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10000, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=20,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load', default=15, help='load file model')
    # parser.add_option('-c', '--load', dest='load', default=False, help='load file model')
    parser.add_option('--train-data', default='data/scale0.5/TrainData')
    parser.add_option('--val-data', default='data/scale0.5/ValData')
    parser.add_option('--checkpoint-dir', default='checkpoints/refine_net')
    parser.add_option('--crop-size', default=100, type='int')
    parser.add_option('--landmark', default=30, type='int', help='0-18, or 30 for random landmarks')
    parser.add_option('--random-crop-range', default=0.5, type='float')
    # parser.add_option('-s', '--scale', dest='scale', type='flolat', default=0.15, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

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
if __name__ == '__main__' and not torch.cuda.is_available():
    raise SystemExit('The legacy refinement pipeline requires CUDA.')
from dataset import CropDataset
from model import UNet, NestedUNet, SCSE_UNet, SCSENestedUNet, DilatedUNet
from utils import  batch, low_bight_filter
from utils import masks_2_labels_maximum, masks_2_labels_coordinate_weight, masks_2_labels_thread
torch.set_default_tensor_type('torch.cuda.FloatTensor')
def train_net(net,
              epochs=5,
              batch_size=10,
              lr=0.1,
              save_cp=True,
              gpu=False,):

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_dataset),
               len(val_dataset), str(save_cp), str(gpu)))


    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=1e-4)
    # optimizer = adabound.AdaBound(model.parameters(), lr=1e-2, final_lr=0.1)
    criterion = nn.BCELoss().cuda()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss(size_average=True).cuda()
    # criterion = nn.NLLLoss()
    criterion1 = FocalLoss()
    # landmarks_loss = nn.MSELoss()
    best_metrics = 10
    loss_history = []
    early_stop_count = 0
    e_loss_best = 100
    best_dice = 0.0001
    for epoch in range(epochs):
        logger.info(f'Epoch: {epoch}')
        logger.info(f'Learning rate: {lr}')
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        train_losses = []

        with tqdm(train_loader) as _tqdm:
            for batched in _tqdm:
                img_region, label_region, masks_region, begin_crop = batched
                # img_region = img_region.type(torch.cuda.FloatTensor)
                # masks_region = masks_region.type(torch.cuda.FloatTensor)
                # if gpu:
                #     img_region, masks_region = img_region.to(device), masks_region.to(device)
                optimizer.zero_grad()

                masks_pred = model(img_region)
                # masks_pred = masks_pred.type(torch.cuda.LongTensor)
                # masks_pred = low_bight_filter(masks_pred)
                # masks_region_one_chan = torch.zeros((10, 250, 250)).type(torch.cuda.FloatTensor)
                # for i in range(10):
                #     for j in range(19):
                #         masks_region_one_chan[i] = masks_region[i][j] * (j + 1) + masks_region_one_chan[i]
                # masks_region_one_chan = masks_region_one_chan.type(torch.cuda.LongTensor)

                loss = criterion(masks_pred, masks_region)
                loss1 = criterion1(masks_pred, masks_region)
                loss = (loss+loss1)/2

                _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}'))
                train_losses.append(loss.item())

                loss.backward()
                optimizer.step()


        # scheduler.step()
        train_loss = np.mean(train_losses)
        # train_iou = np.mean(train_ious)
        logger.info(f'train loss: {train_loss}')
        # logger.info(f'train iou: {train_iou}')
        #
        # valid_losses = []
        # valid_ious = []
        # #model.eval()
        # with torch.no_grad():
        #     with tqdm(val_loader) as _tqdm:
        #         for batched in _tqdm:
        #             img_region, label_region, masks_region, begin_crop = batched
        #
        #             masks_pred = model(img_region)
        #
        #             loss = criterion(masks_pred, masks_region)
        #
        #
        #             _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}'))
        #             valid_losses.append(loss.item())
        # #
        # # # val_dice = eval_net(net, val_loader, gpu)
        # # model.train()
        # valid_loss = np.mean(valid_losses)
        # logger.info(f'valid seg loss: {valid_loss}')

        loss_history.append([train_loss, 0])
        # loss_history.append([train_loss, valid_loss])
        history_ploter(loss_history, log_dir.joinpath('loss.png'))

        torch.save(model.state_dict(), dir_checkpoint+'model_tmp.pth')
        if best_metrics > train_loss:

            best_metrics = train_loss
            logger.info('Best Model! Early stop recount.')
            torch.save(model.state_dict(), dir_checkpoint+'Bestmodel_'+str(epoch)+'.pth')
            early_stop_count = 0
        else:
            early_stop_count = early_stop_count + 1
        logger.info(f'Early stop count: {early_stop_count}')
        history_dict = {'loss': loss_history,
                        'best_metrics': best_metrics}
        with open(log_dir.joinpath('history.pkl'), 'wb') as f:
            pickle.dump(history_dict, f)

        if early_stop_count == 1000:
            print('early stop')
            break

def eval_net(net, dataLoader, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    dataset = dataLoader.dataset
    for i, d in enumerate(dataset):
        img = d[0]
        true_mask = d[1]
        img_location = d[2]


        img = torch.from_numpy(img).unsqueeze(0)
        img = img.type(torch.FloatTensor)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        true_mask = true_mask.type(torch.FloatTensor)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        # mask_pred = (mask_pred > 0.5).float()
        mask_pred = mask_pred.reshape(true_mask.shape)
        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / i


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        raise SystemExit('The legacy refinement pipeline requires a CUDA-enabled PyTorch installation.')
    dir_checkpoint = str(Path(args.checkpoint_dir)) + os.sep
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    run_numbers = [int(p.name) for p in Path(dir_checkpoint).iterdir()
                   if p.is_dir() and p.name.isdigit()]
    folders_num = int(args.load) if args.load else max(run_numbers, default=0) + 1
    dir_checkpoint = str(Path(dir_checkpoint) / f'{folders_num:03d}') + os.sep
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    train_data_path = str(Path(args.train_data)) + os.sep
    val_data_path = str(Path(args.val_data)) + os.sep

    train_dataset = CropDataset(data_path=train_data_path, crop_size=args.crop_size, train_landmark=args.landmark, random_crop_range=args.random_crop_range,
                                mutli_channels=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

    val_dataset = CropDataset(data_path=val_data_path, crop_size=args.crop_size, train_landmark=args.landmark, random_crop_range=args.random_crop_range,
                              mutli_channels=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)

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
        load_dir = dir_checkpoint+'Bestmodel_'+str(last_best_num)+'.pth'
        model.load_state_dict(torch.load(load_dir, weights_only=True))
        print('Model loaded from {}'.format(load_dir))

    if args.gpu:
        model.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=model,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), dir_checkpoint+'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
