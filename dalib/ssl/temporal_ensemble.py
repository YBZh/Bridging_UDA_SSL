from typing import Optional, List, Dict
import torch
import torch.nn as nn
from dalib.modules.classifier import Classifier as ClassifierBase

def consistency_loss(current, temporal):
    loss = ((current - temporal)**2).sum(1).mean()
    return loss

class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256):
        bottleneck = nn.Sequential(
            nn.Dropout(0.5)
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, backbone.out_features)


class ImageClassifierHead(nn.Module):
    r"""Classifier Head for MCD.
    Parameters:
        - **in_features** (int): Dimension of input features
        - **num_classes** (int): Number of classes
        - **bottleneck_dim** (int, optional): Feature dimension of the bottleneck layer. Default: 1024

    Shape:
        - Inputs: :math:`(minibatch, F)` where F = `in_features`.
        - Output: :math:`(minibatch, C)` where C = `num_classes`.
    """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 1024, fc_number: Optional[int] = 1):
        super(ImageClassifierHead, self).__init__()
        if fc_number == 1:
            self.head = nn.Sequential(
                nn.Dropout(0.5),  ### adding one dropout here, since pi-model, temporal ensemble, and mean teacher both need drop to introduce randomness.
                nn.Linear(in_features, num_classes)
            )
        elif fc_number == 2:
            self.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(bottleneck_dim, num_classes)
            )
        elif fc_number == 3:
            self.head = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(bottleneck_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, num_classes)
            )
        else:
            raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs)

    def get_parameters(self) -> List[Dict]:
        """
        :return: A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params