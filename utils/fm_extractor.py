import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    """
    Feature extractor for extracting feature maps from a pretrained model.
    It also changes ReLU to inplace=False (without which we get error)
    and MaxPool to AvgPool which helps propagate gradients better
    """

    def __init__(self, model, layers_to_extract=None):
        super().__init__()
        self.model = model
        self.extract_fm_from_layers = (
            layers_to_extract if layers_to_extract is not None else []
        )

    def forward(self, x):
        fms = []

        for name, layer in self.model.named_children():
            if isinstance(layer, nn.ReLU):
                x = F.relu(x, inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                x = nn.AvgPool2d(2, 2)(x)
            else:
                x = layer(x)

            if name in self.extract_fm_from_layers:
                fms.append((name, x))

        return fms
