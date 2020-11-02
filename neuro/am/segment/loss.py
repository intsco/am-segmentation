import torch
import torch.nn as nn


def soft_dice_jaccard(output, target, jaccard=False, smooth=1.):
    output = torch.sigmoid(output)

    intersection = (output * target).sum()
    union_plus_intersection = output.sum() + target.sum()

    if jaccard:
        return (intersection + smooth) / (union_plus_intersection - intersection + smooth)
    else:
        return (2. * intersection + smooth) / (union_plus_intersection + smooth)


class CombinedLoss(object):
    """ Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard or SoftDice
    """
    def __init__(self, bce_weight=None, jaccard=False, smooth=1.):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.jaccard = jaccard
        self.smooth = smooth

    def __call__(self, output, target):
        loss = self.bce_weight * self.nll_loss(output, target)
        loss += (1 - self.bce_weight) * - torch.log(soft_dice_jaccard(output, target,
                                                                      self.jaccard, self.smooth))
        return loss


def jaccard(y_pred, y_true):
    y_true = torch.squeeze(y_true, dim=1)
    y_pred = torch.squeeze(y_pred, dim=1)
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union_plus_intersection = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)
    return (intersection + epsilon) / (union_plus_intersection - intersection + epsilon)
