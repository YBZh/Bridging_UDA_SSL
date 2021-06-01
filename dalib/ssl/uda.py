from typing import Optional
import torch
import torch.nn as nn

from dalib.modules.classifier import Classifier as ClassifierBase
import torch.nn.functional as F
import ipdb

def kl_div_with_logit(q_logit, p_logit):
    ### return a matrix without mean over samples.
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1)
    qlogp = ( q *logp).sum(dim=1)

    return qlogq - qlogp


def consistency_loss(logits_w, logits_s, target_gt_for_visual, T=1.0, p_cutoff=0.9):
    logits_w = logits_w.detach()
    logits_w = logits_w / T
    logits_s = logits_s / T

    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    # print('max score is: %3f, mean score is: %3f' % (max(max_probs).item(), max_probs.mean().item()))
    mask_binary = max_probs.ge(p_cutoff)
    mask = mask_binary.float()

    masked_loss = kl_div_with_logit(logits_w, logits_s) * mask

    if mask.mean().item() == 0:
        acc_selected = 0
    else:
        acc_selected = (target_gt_for_visual[mask_binary] == max_idx[mask_binary]).float().mean().item()
    return masked_loss.mean(), mask.mean(), acc_selected



