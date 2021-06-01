from typing import Optional
import torch
import torch.nn as nn
from dalib.modules.grl import WarmStartGradientReverseLayer
from dalib.modules.classifier import Classifier as ClassifierBase
import torch.nn.functional as F
import ipdb






#### whether adopt the bottleneck here is to be determined.
class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim)
