import os
import torch
import torch.nn as nn
import shutil
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import ipdb

def extract_acc_from_file(file=None, last_num=20):
    log = open(file, 'r')
    select_acc_list = []
    normal_acc_list = []
    ema_acc_list = []
    for ind, x in enumerate(log.readlines()):
        if 'Select number' in x:
            select_acc = x.split(':')[-1]
            select_acc_list.append(float(select_acc))
        if 'Acc' in x:
            normal_acc_string = x.split(':')[4]
            normal_acc = normal_acc_string.split(',')[0]
            ema_acc = x.split(':')[-1]
            normal_acc_list.append(float(normal_acc))
            ema_acc_list.append(float(ema_acc))

    select_acc = np.median(select_acc_list[-last_num:])
    normal_acc = np.median(normal_acc_list[-last_num:])
    ema_acc = np.median(ema_acc_list[-last_num:])
    print(select_acc, normal_acc, ema_acc)
    return select_acc, normal_acc, ema_acc