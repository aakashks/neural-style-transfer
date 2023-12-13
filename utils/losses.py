import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    def __init__(self, content_fm):
        super().__init__()
        # feature maps of content image
        self.content_fm = content_fm.detach()

    def forward(self, input_fm):
        # same feature maps of the input image
        return F.mse_loss(input_fm, self.content_fm)


class StyleLoss(nn.Module):
    def __init__(self, style_fm):
        super().__init__()
        self.style_fm_gram = self.gram_matrix(style_fm.detach())

    @staticmethod
    def gram_matrix(fm):
        # calculate gram matrix and normalize it
        b, c, h, w = fm.size()
        features = fm.view(b * c, h * w)

        return torch.mm(features, features.t()).div(b * c * h * w)

    def forward(self, input_fm):
        return F.mse_loss(self.gram_matrix(input_fm), self.style_fm_gram)


class TVLoss(nn.Module):
    """
    Calculate total variation Loss
    Basically works as a regularizer
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        return torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff))
