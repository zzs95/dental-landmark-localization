import torch
from torch.autograd import Function, Variable

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter)\
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

# def dice_coef(y_true, y_pred, smooth=1):
#     mean_loss = 0;
#
#     for i in range(y_pred.shape(-1)):
#         intersection = K.sum(y_true[:,:,:,i] * y_pred[:,:,:,i], axis=[1,2,3])
#         union = K.sum(y_true[:,:,:,i], axis=[1,2,3]) + K.sum(y_pred[:,:,:,i], axis=[1,2,3])
#     mean_loss += (2. * intersection + smooth) / (union + smooth)
#     return K.mean(mean_loss, axis=0)
# def dice_coef_loss(y_true, y_pred):
# 	return 1 - dice_coef(y_true, y_pred, smooth=1)

class MultiC_DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        input = input.squeeze()
        target = target.squeeze()
        self.save_for_backward(input, target)
        eps = 0.0001
        channal,_,_ = target.shape
        t_sum = 0
        for i in range(channal):
            self.inter = torch.dot(input[i].view(-1), target[i].view(-1))
            self.union = torch.sum(input[i]) + torch.sum(target[i]) + eps

            t = (2 * self.inter.float() + eps) / self.union.float()
            t_sum = t_sum + t

        return t_sum


def MultiC_dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + MultiC_DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)