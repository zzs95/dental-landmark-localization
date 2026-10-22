import torch
import torch.nn.functional as F

from dice_loss import dice_coeff, MultiC_dice_coeff
from focal_loss import FocalLoss



def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, d in enumerate(dataset):
        img = d[0]
        true_mask = d[1][0]
        true_label = d[1][1]


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
#
# img = img.squeeze()
# fig = plt.figure()
# ax2 = fig.add_subplot(1, 1, 1)
# ax2.imshow(img, cmap="gray")
#
# label_real = true_label.reshape((19, 2))
# label_real = np.array(label_real)
#
# for c in label_real:
# #     # add patches to original image
# #     # could also just plt.scatter() but less control then
#     ax2.add_patch(plt.Circle(c, 5, color='r'))
#     # plt.text(c.asnumpy()[0], c.asnumpy()[1], s=str(i), fontsize=5, color='r')
#     # i=i+1
# plt.show()
# print('ok')

#
# from matplotlib import pyplot as plt
# import numpy as np
#
# pred_mask_np = np.array(mask_pred.detach().float().squeeze())
# for i in range(20):
#     plt.figure()
#     im = pred_mask_np[i]
#     plt.imshow(im, plt.cm.gray)
#     plt.show()
#     plt.imsave(im, 'mask' + str(i) + '.jpg')

# from matplotlib import pyplot as plt
# import numpy as np
#
# pred_mask_np = np.array(mask_pred.detach().float().squeeze())
# for i in range(20):
#     plt.figure()
#     fig = plt.gcf()
#     fig.set_size_inches(5, 5)
#     plt.axis('off')
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     plt.margins(0, 0)
#     im = pred_mask_np[i]
#     plt.imshow(im, plt.cm.gray)
#     if not i == 19:
#         plt.scatter(true_label[i][0], true_label[i][1], color='r', s=20)
#     plt.show()
#     plt.savefig('pred_masks' + str(i) + '.jpg', pad_inches=0)

#
# img = img.squeeze()
# plt.figure()
# fig = plt.gcf()
# fig.set_size_inches(5, 5)
# plt.axis('off')
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# ax2 = fig.add_subplot(1, 1, 1)
# ax2.imshow(img, cmap="gray")
#
# label_real = true_label.reshape((19, 2))
# label_real = np.array(label_real)
#
# for c in label_real:
# #     # add patches to original image
# #     # could also just plt.scatter() but less control then
#     ax2.add_patch(plt.Circle(c, 5, color='r'))
#     # plt.text(c.asnumpy()[0], c.asnumpy()[1], s=str(i), fontsize=5, color='r')
#     # i=i+1
# plt.show()
# plt.savefig('pred_masks' + str(i) + '.jpg', pad_inches=0)
# print('ok')

# pred_mask_np = np.array(mask_pred.detach().float().squeeze())
# for i in range(20):
#     plt.figure()
#     fig = plt.gcf()
#     fig.set_size_inches(5, 5)
#     plt.axis('off')
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#     plt.margins(0, 0)
#     img_np = img.cpu().numpy().squeeze()
#
#     if not i == 19:
#         im = pred_mask_np[i] / pred_mask_np[i].max() * img_np
#         plt.imshow(im, plt.cm.gray)
#         plt.scatter(true_label[i][0], true_label[i][1], color='r', s=20)
#     if i ==19:
#         im = (np.ones(img_np.shape)-pred_mask_np[i] / pred_mask_np[i].max()) * (img_np)
#         plt.imshow(im, plt.cm.gray)
#         for j in range(19):
#             plt.scatter(true_label[j][0], true_label[j][1], color='r', s=20)
#     plt.show()
#     plt.savefig('img_mask' + str(i) + '.jpg', pad_inches=0)
