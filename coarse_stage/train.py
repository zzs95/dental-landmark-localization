from optparse import OptionParser

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5000, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load', default=None,
                      help='optional checkpoint state_dict to resume')
    parser.add_option('--data-root', default='data/Cephalometry_crop',
                      help='prepared cropped images and 400_ave annotations')
    parser.add_option('--checkpoint-dir', default='checkpoints/stage1')
    parser.add_option('--log-dir', default='runs/stage1')
    parser.add_option('--cpu', action='store_false', dest='gpu')
    parser.add_option('-s', '--scale', dest='scale', type='float', default=0.15, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

import sys
# sys.path
import os
from optparse import OptionParser
import numpy as np
from dice_loss import dice_coeff
from focal_loss import FocalLoss

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from eval import eval_net
from unet import UNet, NestedUNet, SCSE_UNet, SCSENestedUNet, DilatedUNet
from utils import get_ids, split_ids, split_shuffle_data, get_imgs_and_masks, batch, masks_2_labels_maximum, masks_2_labels_coordinate_weight, masks_2_labels_thread, masks_2_labels_normal_thread, masks_2_labels_max_tread
torch.set_default_tensor_type('torch.FloatTensor')
def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              # val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.1,
              data_root="data/Cephalometry_crop",
              checkpoint_dir="checkpoints/stage1",
              log_dir="runs/stage1"):

    from pathlib import Path
    root = Path(data_root)
    dir_checkpoint = str(Path(checkpoint_dir)) + os.sep
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    dir_train_img = str(root / 'RawImage' / 'TrainData') + os.sep
    dir_test_img = str(root / 'RawImage' / 'TestData') + os.sep
    dir_val_img = str(root / 'RawImage' / 'ValData') + os.sep
    dir_label = str(root / 'AnnotationsByMD' / '400_ave') + os.sep
    for directory in (dir_train_img, dir_test_img, dir_val_img, dir_label):
        if not Path(directory).is_dir():
            raise FileNotFoundError(f'Missing prepared dataset directory: {directory}')

    ids_train_img = get_ids(dir_train_img)
    ids_test_img = get_ids(dir_test_img)
    ids_val_img = get_ids(dir_val_img)
    # ids = split_ids(ids)

    train_dataset = split_shuffle_data(ids_train_img,  shuffle=True)
    test_dataset = split_shuffle_data(ids_test_img,  shuffle=False)
    val_dataset = split_shuffle_data(ids_val_img,  shuffle=False)

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
               len(test_dataset), str(save_cp), str(gpu)))

    N_train = len(train_dataset)

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=1e-4)

    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss().cuda()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    criterion1 = FocalLoss()
    # landmarks_loss = nn.MSELoss()
    early_stop_count = 0
    e_loss_best = 100
    best_dice = 0.0001
    all_loss = list([])
    all_loss0 = list([])
    all_loss1 = list([])
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(train_dataset, dir_train_img, dir_label, img_scale, data_aug=False)
        # val = get_imgs_and_masks(val_dataset, dir_val_img, dir_label, img_scale, data_aug=False)
        test = get_imgs_and_masks(test_dataset, dir_test_img, dir_label, img_scale, data_aug=False)

        epoch_loss = 0
        epoch_loss0 = 0
        epoch_loss1 = 0


        val_dice = eval_net(net, test, gpu)
        net.train()

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float16)
            true_masks = np.array([i[1][0] for i in b]).astype(np.float16)
            # true_masks_labels = np.array([i[1][1] for i in b]).astype(np.float16)

            imgs = torch.from_numpy(imgs)
            imgs = imgs.type(torch.FloatTensor)
            true_masks = torch.from_numpy(true_masks)
            true_masks = true_masks.type(torch.FloatTensor)


            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            # masks_probs_flat = masks_pred.view(-1)
            # true_masks_flat = true_masks.view(-1)
            loss0 = criterion(masks_pred, true_masks)
            loss1 = criterion1(masks_pred, true_masks)
            loss = (loss0+loss1)/2
            # loss = criterion1(masks_probs_flat, true_masks_flat)
            # print('focal_loss,'+str(loss)+',bce_loss '+str(loss1))
            # loss = (loss + loss1)/2

            # # label loss
            # img_width = imgs.shape[3]
            # pred_masks_labels = masks_2_labels_maximum(masks_pred).astype('float16')
            # pred_masks_labels = torch.from_numpy(pred_masks_labels/img_width).type(torch.FloatTensor)
            # true_masks_labels = torch.from_numpy(true_masks_labels/img_width).type(torch.FloatTensor)
            # loss_2 = landmarks_loss(true_masks_labels, pred_masks_labels/img_width)
            # loss_2 = torch.from_numpy(np.array(loss_2)).type(torch.cuda.FloatTensor)
            #
            # lamda = np.array(0.001)
            # lamda = torch.from_numpy(lamda).type(torch.cuda.FloatTensor)
            # loss = loss+loss_2*lamda
            #
            # # dice loss
            # train_dice = dice_coeff(masks_pred, true_masks).item()/batch_size
            # loss_3 = 1 - train_dice
            # loss = (loss + loss_3)/2

            epoch_loss += loss.item()
            epoch_loss0 += loss0.item()
            epoch_loss1 += loss1.item()

            # print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        e_loss = epoch_loss / (i + 1)
        e_loss0 = epoch_loss0 / (i + 1)
        e_loss1 = epoch_loss1 / (i + 1)
        all_loss.append(e_loss)
        all_loss0.append(e_loss0)
        all_loss1.append(e_loss1)
        all_loss_np = np.array(all_loss)
        np.save(os.path.join(log_dir, 'MEANLoss.npy'), all_loss_np)
        all_loss_np = np.array(all_loss0)
        np.save(os.path.join(log_dir, 'BCELoss.npy'), all_loss_np)
        all_loss_np = np.array(all_loss1)
        np.save(os.path.join(log_dir, 'FOCALloss.npy'), all_loss_np)
        print('Epoch finished ! Loss: {}'.format(e_loss))


        # val_dice = eval_net(net, val, gpu)
        # print('Validation Dice Coeff: {}'.format(val_dice))


        # if save_cp and epoch > 0 and e_loss < e_loss_best:
        # if save_cp and epoch > 0 and val_dice > best_dice and early_stop_count < 100:
        if save_cp and epoch > 0 and e_loss < e_loss_best and early_stop_count < 100:
        # if save_cp and epoch > 0 :
        #     best_dice = val_dice
            e_loss_best = e_loss
            early_stop_count = 0
            print('update best val_dice, early stop count reset')
            torch.save(net.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            # print('Checkpoint {} saved !'.format(epoch + 1))
        else:
            early_stop_count = early_stop_count+1
            print('best val_dice unchange, early stop count '+str(early_stop_count))

        if early_stop_count == 100:
            print('early stop ,save last model '+str(epoch+1))
            torch.save(net.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            break






if __name__ == '__main__':
    args.gpu = args.gpu and torch.cuda.is_available()

    net = UNet(n_channels=1, n_classes=20)
    # net = NestedUNet(n_channels=1, n_classes=20)
    # net = SCSE_UNet(n_channels=1, n_classes=20)
    # net = SCSENestedUNet(n_channels=1, n_classes=20)
    # net = DilatedUNet(in_channels=1, classes=20)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location='cuda' if args.gpu else 'cpu', weights_only=True))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale,
                  data_root=args.data_root,
                  checkpoint_dir=args.checkpoint_dir,
                  log_dir=args.log_dir)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(args.checkpoint_dir, 'INTERRUPTED.pth'))
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
