
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import numpy as np
from typing import Optional, List, Dict
import copy

def entropy_including_softmax(logit: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logit, dim=1)
    return -(p * F.log_softmax(logit, dim=1)).sum(dim=1).mean(dim=0)


def kl_div_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = (q * logq).sum(dim=1).mean(dim=0)
    qlogp = (q * logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp


def _l2_normalize(d):
    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


def vat_loss(classifier, ul_x, ul_y, xi=1e-6, eps=1.0, num_iters=1):
    # find r_adv
    d = torch.Tensor(ul_x.size()).normal_()

    classifier_copy = copy.deepcopy(classifier)

    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        d = d.cuda().requires_grad_()
        y_hat, _ = classifier_copy(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()
        d = d.grad.clone().cpu()
        classifier_copy.zero_grad()
    d = _l2_normalize(d)
    d = d.cuda()
    r_adv = eps * d

    # classifier.load_state_dict(org_state_G)
    # F.load_state_dict(org_state_F)
    # compute lds

    y_hat, _ = classifier(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    del classifier_copy
    return delta_kl

class Feature_extractor(nn.Module):
    """A generic feature extractor class for domain adaptation.

    Parameters:
        - **backbone** (class:`nn.Module` object): Any backbone to extract 1-d features from data
        - **bottleneck** (class:`nn.Module` object, optional): Any bottleneck layer. Use no bottleneck by default
        - **bottleneck_dim** (int, optional): Feature dimension of the bottleneck layer. Default: -1


    .. note::
        Different classifiers are used in different domain adaptation algorithms to achieve better accuracy
        respectively, and we provide a suggested `Classifier` for different algorithms.
        Remember they are not the core of algorithms. You can implement your own `Classifier` and combine it with
        the domain adaptation algorithm in this algorithm library.

    .. note::
        The learning rate of this classifier is set 10 times to that of the feature extractor for better accuracy
        by default. If you have other optimization strategies, please over-ride `get_parameters`.

    Inputs:
        - **x** (tensor): input data fed to `backbone`

    Outputs: predictions, features
        - **features**: features after `bottleneck` layer

    Shape:
        - Inputs: (minibatch, *) where * means, any number of additional dimensions
        - predictions: (minibatch, `num_classes`)
        - features: (minibatch, `features_dim`)

    """

    def __init__(self, backbone: nn.Module, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1):
        super(Feature_extractor, self).__init__()
        self.backbone = backbone

        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim


    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        f = self.backbone(x)
        f = f.view(-1, self.backbone.out_features)
        f = self.bottleneck(f)

        return f

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
        ]
        return params


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
            {"params": self.head.parameters(),  "lr_mult": 1.},
        ]
        return params
