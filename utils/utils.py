import os
import torch
import torch.nn as nn
import shutil
import torch.nn.functional as F
from sklearn.cluster import KMeans
# from spherecluster import SphericalKMeans
import numpy as np
import random
import ipdb

def get_random_recover_index(num):
    list_rand = list(range(num))
    random.shuffle(list_rand)

    list_recover = [0] * num
    for normal_index in range(num):
        list_recover[list_rand[normal_index]] = normal_index

    return list_rand, list_recover

def cg_solver(A, B, X0=None, rtol=1e-3, maxiter=None):
    n, m = B.shape
    if X0 is None:
        X0 = B
    if maxiter is None:
        maxiter = 2 * min(n, m)
    X_k = X0
    R_k = B - A.matmul(X_k)
    P_k = R_k
    stopping_matrix = torch.max(rtol * torch.abs(B), 1e-3 * torch.ones_like(B))
    for k in range(1, maxiter+1):
        fenzi = R_k.transpose(0,1).matmul(R_k).diag()
        fenmu = P_k.transpose(0,1).matmul(A).matmul(P_k).diag()
        #fenmu[fenmu == 0] = 1e-8
        alpha_k = fenzi / fenmu
        X_kp1 = X_k + alpha_k * P_k
        R_kp1 = R_k - (A.matmul(alpha_k * P_k))
        residual_norm = torch.abs(A.matmul(X_kp1) - B)
        if (residual_norm <= stopping_matrix).all():
            break
        #fenzi[fenzi ==0] = 1e-8
        beta_k = (R_kp1.transpose(0, 1).matmul(R_kp1) / (fenzi)).diag()
        P_kp1 = R_kp1 + beta_k * P_k

        P_k = P_kp1
        X_k = X_kp1
        R_k = R_kp1
    return X_kp1

def get_prediction_with_uniform_prior(soft_prediction):

    soft_prediction_uniform = soft_prediction / soft_prediction.sum(0, keepdim=True).pow(0.5)
    soft_prediction_uniform /= soft_prediction_uniform.sum(1, keepdim=True)
    return soft_prediction_uniform


def get_labels_from_classifier_prediction(target_u_prediction_matrix, T, gt_label):
    target_u_prediction_matrix_withT = target_u_prediction_matrix / T
    soft_label_fc = torch.softmax(target_u_prediction_matrix_withT, dim=1)
    scores, hard_label_fc = torch.max(soft_label_fc, dim=1)

    soft_label_uniform_fc = get_prediction_with_uniform_prior(soft_label_fc)
    scores_uniform, hard_label_uniform_fc = torch.max(soft_label_fc, dim=1)

    acc_fc = accuracy(soft_label_fc, gt_label)
    acc_uniform_fc = accuracy(soft_label_uniform_fc, gt_label)
    print('acc of fc is: %3f' % (acc_fc))
    print('acc of fc with uniform prior is: %3f' % (acc_uniform_fc))


    return soft_label_fc, soft_label_uniform_fc, hard_label_fc, hard_label_uniform_fc, acc_fc, acc_uniform_fc


def get_labels_from_Sphericalkmeans(initial_centers_array, target_u_feature, num_class, gt_label, T=0.05, max_iter=100, target_l_feature=None):
    raise NotImplementedError  ### SphericalKMeans is confict with current edition
#     ## initial_centers: num_cate * feature dim
#     ## target_u_feature: num_u * feature dim
#     if type(target_l_feature) == torch.Tensor: ### if there are some labeled data of the same domain
#         target_u_feature_array = torch.cat((target_u_feature, target_l_feature), dim=0).numpy()
#     else:
#         target_u_feature_array = target_u_feature.numpy()
#     kmeans = SphericalKMeans(n_clusters=num_class, random_state=0, init=initial_centers_array,
#                              max_iter=max_iter).fit(target_u_feature_array)
#     Ind = kmeans.labels_
#     Ind_tensor = torch.from_numpy(Ind)
#     centers = kmeans.cluster_centers_  ### num_category * feature_dim
#     centers_tensor = torch.from_numpy(centers)
#
#     cos_similarity = torch.matmul(target_u_feature, centers_tensor.transpose(0, 1))
#     soft_label_kmeans = torch.softmax(cos_similarity / T, dim=1)
#     scores, hard_label_kmeans = torch.max(soft_label_kmeans, dim=1)
#
#     #ipdb.set_trace() #### check the difference between Ind_tensor and hard_label_kmeans
#
#     soft_label_uniform_kmeans = get_prediction_with_uniform_prior(soft_label_kmeans)
#     scores_uniform, hard_label_uniform_kmeans = torch.max(soft_label_kmeans, dim=1)
#
#     acc_kmeans = accuracy(soft_label_kmeans, gt_label)
#     acc_uniform_kmeans = accuracy(soft_label_uniform_kmeans, gt_label)
#     print('acc of sphe-kmeans is: %3f' % (acc_kmeans))
#     print('acc of sphe-kmeans with uniform prior is: %3f' % (acc_uniform_kmeans))
#
#
#     return soft_label_kmeans, soft_label_uniform_kmeans, hard_label_kmeans, hard_label_uniform_kmeans, acc_kmeans, acc_uniform_kmeans



def get_labels_from_lp(labeled_features, labeled_onehot_gt, unlabeled_features, gt_label, num_class, dis='cos', solver='closedform', graphk=20, alpha=0.75):

    num_labeled = labeled_features.size(0)
    if num_labeled > 100000:
        print('too many labeled data, randomly select a subset')
        indices = torch.randperm(num_labeled)[:10000]
        labeled_features = labeled_features[indices]
        labeled_onehot_gt  = labeled_onehot_gt[indices]
        num_labeled = 10000

    num_unlabeled = unlabeled_features.size(0)
    num_all = num_unlabeled + num_labeled
    all_features = torch.cat((labeled_features, unlabeled_features), dim=0)
    unlabeled_zero_gt = torch.zeros(num_unlabeled, num_class)
    all_gt = torch.cat((labeled_onehot_gt, unlabeled_zero_gt), dim=0)
    ### calculate the affinity matrix
    if dis == 'cos':
        all_features = F.normalize(all_features, dim=1, p=2)
        weight = torch.matmul(all_features, all_features.transpose(0, 1))
        weight[weight < 0] = 0
        values, indexes = torch.topk(weight, graphk)
        weight[weight < values[:, -1].view(-1, 1)] = 0
        weight = weight + weight.transpose(0, 1)

    elif dis == 'l2':
        all_features_unsq1 = torch.unsqueeze(all_features, 1)
        all_features_unsq2 = torch.unsqueeze(all_features, 0)
        weight = ((all_features_unsq1 - all_features_unsq2) ** 2).mean(2)
        #weight = torch.exp( - weight / 2.0)
        weight = 1.0 / (weight + 1e-8)
        values, indexes = torch.topk(weight, graphk)
        weight[weight < values[:, -1].view(-1, 1)] = 0
        weight = weight + weight.transpose(0, 1)
    elif dis == 'nndescent':
        from pynndescent import NNDescent
        numpy_data = all_features.numpy()
        weight = torch.zeros(all_features.size(0), all_features.size(0))
        knn_indices, knn_value = NNDescent(numpy_data, "cosine", {}, graphk, random_state=np.random)._neighbor_graph
        weight.scatter_(1, torch.from_numpy(knn_indices), 1 - torch.from_numpy(knn_value).float())
        weight = weight + torch.t(weight)
    else:
        raise NotImplementedError
    weight.diagonal(0).fill_(0)  ## change the diagonal elements with inplace operation.
    if solver == 'closedform':
        D = weight.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-8))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, num_all)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(num_all, 1)
        S = D1 * weight * D2  ############ same with D3 = torch.diag(D_sqrt_inv)  S = torch.matmul(torch.matmul(D3, weight), D3)
        pred_all = torch.matmul(torch.inverse(torch.eye(num_all) - alpha * S + 1e-8), all_gt)
    elif solver == 'CG':
        D = weight.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-8))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, num_all)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(num_all, 1)
        S = D1 * weight * D2
        A = torch.eye(num_all) - alpha * S + 1e-8
        pred_all = cg_solver(A, all_gt)
    else:
        raise NotImplementedError
    del weight
    pred_unl = pred_all[num_labeled:, :]
    #### add a fix value
    min_value = torch.min(pred_unl, 1)[0]
    min_value[min_value > 0] = 0
    pred_unl = pred_unl - min_value.view(-1, 1)

    pred_unl = pred_unl / pred_unl.sum(1).view(-1, 1)

    soft_label_lp = pred_unl
    scores, hard_label_lp = torch.max(soft_label_lp, dim=1)

    soft_label_uniform_lp = get_prediction_with_uniform_prior(soft_label_lp)
    scores_uniform, hard_label_uniform_lp = torch.max(soft_label_lp, dim=1)

    acc_lp = accuracy(soft_label_lp, gt_label)
    acc_uniform_lp = accuracy(soft_label_uniform_lp, gt_label)
    print('acc of lp is: %3f' % (acc_lp))
    print('acc of lp with uniform prior is: %3f' % (acc_uniform_lp))


    return soft_label_lp, soft_label_uniform_lp, hard_label_lp, hard_label_uniform_lp, acc_lp, acc_uniform_lp

def get_labels_from_kmeans(initial_centers_array, target_u_feature, num_class, gt_label, T=1.0, max_iter=100, target_l_feature=None):
    ## initial_centers: num_cate * feature dim
    ## target_u_feature: num_u * feature dim
    if type(target_l_feature) == torch.Tensor:  ### if there are some labeled data of the same domain
        target_u_feature_array = torch.cat((target_u_feature, target_l_feature), dim=0).numpy()
    else:
        target_u_feature_array = target_u_feature.numpy()
    #target_u_feature_array = target_u_feature.numpy()
    kmeans = KMeans(n_clusters=num_class, random_state=0, init=initial_centers_array,
                             max_iter=max_iter).fit(target_u_feature_array)
    Ind = kmeans.labels_
    Ind_tensor = torch.from_numpy(Ind)
    centers = kmeans.cluster_centers_  ### num_category * feature_dim
    centers_tensor = torch.from_numpy(centers)

    centers_tensor_unsq = torch.unsqueeze(centers_tensor, 0)
    target_u_feature_unsq = torch.unsqueeze(target_u_feature, 1)
    L2_dis = ((target_u_feature_unsq - centers_tensor_unsq)**2).mean(2)
    soft_label_kmeans = torch.softmax(1 + 1.0 / (L2_dis + 1e-8), dim=1)
    # cos_similarity = torch.matmul(target_u_feature, centers_tensor.transpose(0, 1))
    # soft_label_kmeans = torch.softmax(cos_similarity / T, dim=1)
    scores, hard_label_kmeans = torch.max(soft_label_kmeans, dim=1)


    soft_label_uniform_kmeans = get_prediction_with_uniform_prior(soft_label_kmeans)
    scores_uniform, hard_label_uniform_kmeans = torch.max(soft_label_kmeans, dim=1)

    acc_kmeans = accuracy(soft_label_kmeans, gt_label)
    acc_uniform_kmeans = accuracy(soft_label_uniform_kmeans, gt_label)
    print('acc of kmeans is: %3f' % (acc_kmeans))
    print('acc of kmens with uniform prior is: %3f' % (acc_uniform_kmeans))


    return soft_label_kmeans, soft_label_uniform_kmeans, hard_label_kmeans, hard_label_uniform_kmeans, acc_kmeans, acc_uniform_kmeans

class LabelGuessor(object):

    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, G, F, ims):
        org_state_G = {
            k: v.clone().detach()
            for k, v in G.state_dict().items()
        }
        org_state_F = {
            k: v.clone().detach()
            for k, v in F.state_dict().items()
        }
        is_train = G.training
        with torch.no_grad():
            G.train()
            F.train()
            all_probs = []
            logits = F(G(ims))
            probs = torch.softmax(logits, dim=1)
            scores, lbs = torch.max(probs, dim=1)
            idx = scores > self.thresh
            lbs = lbs[idx]

        G.load_state_dict(org_state_G)
        F.load_state_dict(org_state_F)
        if is_train:
            G.train()
            F.train()
        else:
            G.eval()
            F.eval()
        return lbs.detach(), idx


class LabelGuessorWithM(object):

    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, G, M, F, ims):
        org_state_G = {
            k: v.clone().detach()
            for k, v in G.state_dict().items()
        }
        org_state_F = {
            k: v.clone().detach()
            for k, v in F.state_dict().items()
        }
        is_train = G.training
        with torch.no_grad():
            G.train()
            F.train()
            all_probs = []
            logits = F(M(G(ims)))
            probs = torch.softmax(logits, dim=1)
            scores, lbs = torch.max(probs, dim=1)
            idx = scores > self.thresh
            lbs = lbs[idx]

        G.load_state_dict(org_state_G)
        F.load_state_dict(org_state_F)
        if is_train:
            G.train()
            F.train()
        else:
            G.eval()
            F.eval()
        return lbs.detach(), idx

class LabelGuessorMMD(object):

    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, G, F, ims):
        org_state_G = {
            k: v.clone().detach()
            for k, v in G.state_dict().items()
        }
        org_state_F = {
            k: v.clone().detach()
            for k, v in F.state_dict().items()
        }
        is_train = G.training
        with torch.no_grad():
            G.train()
            F.train()
            all_probs = []
            _, logits = F(G(ims))
            probs = torch.softmax(logits, dim=1)
            scores, lbs = torch.max(probs, dim=1)
            idx = scores > self.thresh
            lbs = lbs[idx]

        G.load_state_dict(org_state_G)
        F.load_state_dict(org_state_F)
        if is_train:
            G.train()
            F.train()
        else:
            G.eval()
            F.eval()
        return lbs.detach(), idx


class LabelGuessorProto(object):

    def __init__(self, thresh, dis, type='cla'):
        self.thresh = thresh
        self.dis = dis
        self.type = type

    def __call__(self, G, F, ims, proto):
        org_state_G = {
            k: v.clone().detach()
            for k, v in G.state_dict().items()
        }
        org_state_F = {
            k: v.clone().detach()
            for k, v in F.state_dict().items()
        }
        is_train = G.training
        with torch.no_grad():
            G.train()
            F.train()
            feat = G(ims)
            logits = F(feat)
            probs = torch.softmax(logits, dim=1)
            scores, lbs = torch.max(probs, dim=1)

            probs_proto = proto_prob_cal(feat, proto, self.dis)
            scores_proto, lbs_proto = torch.max(probs_proto, dim=1)
            print('prob_classifier_unl', scores.mean())
            print('prob_proto_unl', scores_proto.mean())
            print((lbs == lbs_proto)[:20])
            if self.type == 'cla':
                idx = scores > self.thresh
            elif self.type == 'prot':
                idx = scores_proto > self.thresh
            elif self.type == 'cla_c':
                idx = (scores > self.thresh) & (lbs == lbs_proto)
            elif self.type == 'prot_c':
                idx = (scores_proto > self.thresh) & (lbs == lbs_proto)
            elif self.type == 'both':
                idx = (scores > self.thresh) & (scores_proto > self.thresh) & (lbs == lbs_proto)
            else:
                raise NotImplementedError

            #idx = (scores > self.thresh) & (lbs == lbs_proto)
            lbs = lbs[idx]

        G.load_state_dict(org_state_G)
        F.load_state_dict(org_state_F)
        if is_train:
            G.train()
            F.train()
        else:
            G.eval()
            F.eval()
        return lbs.detach(), idx

def proto_prob_cal(feat, proto, dis='cos'):
    if dis == 'cos':
        proto_norm = F.normalize(proto, dim=1, p=2)
        mul_sim = torch.matmul(feat, proto_norm.transpose(0,1))
        probs = torch.softmax(mul_sim, dim=1)
        # ipdb.set_trace()
        # feat_norm = F.normalize(feat, dim=1, p=2)
        # proto_norm = F.normalize(proto, dim=1, p=2)
        # cos_sim = torch.matmul(feat_norm, proto_norm.transpose(0,1))
        # cos_sim_pro = torch.exp(cos_sim) - 1
        # probs = cos_sim_pro / cos_sim_pro.sum(1).view(-1, 1)
    elif dis == 'l2':
        feat_unsq = torch.unsqueeze(feat, 1)
        proto_unsq = torch.unsqueeze(proto, 0)
        dis = torch.sqrt(((feat_unsq - proto_unsq)**2).sum(2))
        # probs = torch.softmax(1.0 / (dis + 1e-8) * 30 + 1.0, dim=1)
        probs = torch.softmax(-dis, dim=1)
    elif dis == 'mul':
        # mul_sim = torch.matmul(feat, proto.transpose(0, 1))
        # probs = torch.softmax(mul_sim, dim=1)

        # feat_norm = F.normalize(feat, dim=1, p=2)
        # proto_norm = F.normalize(proto, dim=1, p=2)
        mul_sim = torch.matmul(feat, proto.transpose(0, 1)) / 100   ###神奇。。我也不知道为什么这里会这样。
        probs = torch.softmax(mul_sim, dim=1)

    else:
        raise NotImplementedError
    return probs



class LabelGuessorDomainFilter(object):

    def __init__(self, score_thresh, domain_threshold):
        self.thresh = score_thresh
        self.domain_thresh = domain_threshold
        self.mean_domain = 0.5

    def __call__(self, G, F, ims, domain_score):
        adopted_domain_thresh = min(self.domain_thresh, self.mean_domain)
        org_state_G = {
            k: v.clone().detach()
            for k, v in G.state_dict().items()
        }
        org_state_F = {
            k: v.clone().detach()
            for k, v in F.state_dict().items()
        }
        is_train = G.training
        with torch.no_grad():
            G.train()
            F.train()
            all_probs = []
            logits = F(G(ims))
            probs = torch.softmax(logits, dim=1)
            scores, lbs = torch.max(probs, dim=1)

            idx = (scores > self.thresh) & (domain_score[:,0] > adopted_domain_thresh)  ### two criterion
            lbs = lbs[idx]

        G.load_state_dict(org_state_G)
        F.load_state_dict(org_state_F)
        if is_train:
            G.train()
            F.train()
        else:
            G.eval()
            F.eval()

        self.mean_domain = self.mean_domain * 0.7 + domain_score.mean().item() * 0.3
        print('mean domain', self.mean_domain)
        return lbs.detach(), idx


class LabelGuessorDomainFilterFast(object):

    def __init__(self, score_thresh, domain_threshold):
        self.thresh = score_thresh
        self.domain_thresh = domain_threshold
        self.mean_domain = 0.5

    def __call__(self, logits, domain_score):
        adopted_domain_thresh = self.domain_thresh

        probs = torch.softmax(logits, dim=1)
        scores, lbs = torch.max(probs, dim=1)

        idx = (scores > self.thresh) & (domain_score[:,0] > adopted_domain_thresh)  ### two criterion
        lbs = lbs[idx]

        self.mean_domain = self.mean_domain * 0.7 + domain_score.mean().item() * 0.3
        print('mean domain', self.mean_domain)
        return lbs.detach(), idx


class EMA_fixmatch(object):
    def __init__(self, G, F, alpha):
        self.step = 0
        self.G = G
        self.F = F
        self.alpha = alpha
        self.shadow_G = self.get_model_state_G()
        self.shadow_F = self.get_model_state_F()

        self.backup_G = {}
        self.backup_F = {}

        self.param_keys_G = [k for k, _ in self.G.named_parameters()]   ###weight bias
        self.buffer_keys_G = [k for k, _ in self.G.named_buffers()]     ### running mean, var
        self.param_keys_F = [k for k, _ in self.F.named_parameters()]   ###weight bias
        self.buffer_keys_F = [k for k, _ in self.F.named_buffers()]     ### running mean, var

    def update_params(self):  ############### only the shadown is updated.
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state_G = self.G.state_dict()
        for name in self.param_keys_G:
            self.shadow_G[name].copy_(
                decay * self.shadow_G[name]
                + (1 - decay) * state_G[name]
            )
        state_F = self.F.state_dict()
        for name in self.param_keys_F:
            self.shadow_F[name].copy_(
                decay * self.shadow_F[name]
                + (1 - decay) * state_F[name]
            )
        #  for name in self.buffer_keys:
        #      self.shadow[name].copy_(
        #          decay * self.shadow[name]
        #          + (1 - decay) * state[name]
        #      )
        self.step += 1

    def update_buffer(self):
        state_G = self.G.state_dict()
        for name in self.buffer_keys_G:
            self.shadow_G[name].copy_(state_G[name])

        state_F = self.F.state_dict()
        for name in self.buffer_keys_F:
            self.shadow_F[name].copy_(state_F[name])

    def apply_shadow(self):  #### 使用shadow进行评估
        self.backup_G = self.get_model_state_G()
        self.G.load_state_dict(self.shadow_G)

        self.backup_F = self.get_model_state_F()
        self.F.load_state_dict(self.shadow_F)

    def restore(self):
        self.G.load_state_dict(self.backup_G)
        self.F.load_state_dict(self.backup_F)

    def get_model_state_G(self):
        return {
            k: v.clone().detach()
            for k, v in self.G.state_dict().items()
        }
    def get_model_state_F(self):
        return {
            k: v.clone().detach()
            for k, v in self.F.state_dict().items()
        }


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.momentum = 0.0  #### fix all the imagenet pre-trained running mean and average
        # m.eval()

def release_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.momentum = 0.1  #### roll back to the default setting
        # m.train()

class weight_ema(object):
    def __init__(self, G, F, ema_G, ema_F, alpha=0.999, wd=0.0002):
        self.G = G
        self.F = F
        self.ema_G = ema_G
        self.ema_F = ema_F
        self.alpha = alpha
        self.G_params = list(G.state_dict().values())
        self.F_params = list(F.state_dict().values())
        self.ema_G_params = list(ema_G.state_dict().values())
        self.ema_F_params = list(ema_F.state_dict().values())
        self.wd = wd

        for param, ema_param in zip(self.G_params, self.ema_G_params):
            param.data.copy_(ema_param.data)
        for param, ema_param in zip(self.F_params, self.ema_F_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.G_params, self.ema_G_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

        for param, ema_param in zip(self.F_params, self.ema_F_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_cpu(x):
    return x.cpu()

def to_numpy(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_onehot(label, num_classes):
    identity = torch.eye(num_classes).to(label.device)
    onehot = torch.index_select(identity, 0, label)
    return onehot

def accuracy(output, target):
    """Computes the precision"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct = correct[:1].view(-1).float().sum(0, keepdim=True)
    res = correct.mul_(100.0 / batch_size)
    return res


def accuracy_for_each_class(output, target, total_vector, correct_vector):
    """Computes the precision for each class"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1)).float().cpu().squeeze()
    for i in range(batch_size):
        total_vector[target[i]] += 1
        correct_vector[torch.LongTensor([target[i]])] += correct[i]

    return total_vector, correct_vector




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


def save_checkpoint(state, is_best, checkpoint='checkpoint',
                    filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
