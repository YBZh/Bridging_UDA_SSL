## we conduct the DA methods following https://github.com/thuml/Transfer-Learning-Library, which gives better results than the common settings.

import torch
import os
import math
import time
import numpy as np
from .base_solver import BaseSolver
import torch.nn.functional as F
from utils.utils import AverageMeter, to_cuda, accuracy, accuracy_for_each_class, get_random_recover_index



class Solver(BaseSolver):
    def __init__(self, classifier, dataloaders, args, **kwargs):
        super(Solver, self).__init__(classifier, dataloaders, args, **kwargs)
        self.total_iters = self.args.epochs * self.args.iters_per_epoch
        self.ema_m = args.ema_m
        self.gamma = 0.001
        self.decay_rate = 0.75
        self.build_optimizer()
        self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
        self.train_data['target']['iterator'] = iter(self.train_data['target']['loader'])
        assert not args.strongaug  ## StrongAug is NOT expected.

    def update_network(self, **kwargs):
        self.classifier.train()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses_all = AverageMeter()
        losses_cls = AverageMeter()
        losses_ssl = AverageMeter()
        prec1 = AverageMeter()
        end = time.time()
        for i in range(self.iters_per_epoch):
            if self.args.lr_schedule == 'dalib':
                self.lr_sheduler_f.step()
                self.lr_sheduler_g.step()
            elif self.args.lr_schedule == 'inv':
                self.lr_sheduler_f.step(self.iters)
                self.lr_sheduler_g.step(self.iters)
            coeff = 2 / (1 + math.exp(-1 * 10 * self.iters / self.total_iters)) - 1

            (source_data, _), source_gt = self.get_samples('source')
            (target_data_weak, target_data_weak2), target_gt_for_visual = self.get_samples('target')
            data_time.update(time.time() - end)
            ##
            source_data = source_data.cuda(self.args.gpu, non_blocking=True)
            target_data_weak = target_data_weak.cuda(self.args.gpu, non_blocking=True)
            target_data_weak2 = target_data_weak2.cuda(self.args.gpu, non_blocking=True)
            # target_gt_for_visual = target_gt_for_visual.cuda(self.args.gpu, non_blocking=True)
            source_gt_onehot = torch.zeros(source_data.size(0), self.num_classes).scatter_(1, source_gt.view(-1,1).long(), 1)
            source_gt_onehot = source_gt_onehot.cuda(self.args.gpu, non_blocking=True)
            source_gt = source_gt.cuda(self.args.gpu, non_blocking=True)

            with torch.no_grad():
                ## compute guessed labels for unlabeled data
                outputs_u, _ = self.classifier(target_data_weak)
                outputs_u2, _ = self.classifier(target_data_weak2)
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                pt = p ** (1 / self.args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            all_inputs = torch.cat((source_data, target_data_weak, target_data_weak2), dim=0)
            all_targets = torch.cat((source_gt_onehot, targets_u, targets_u), dim=0)
            l = np.random.beta(0.75, 0.75)
            l = max(l, 1-l)
            idx = torch.randperm(all_inputs.size(0))
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            num_s = source_data.size(0)
            if self.args.regular_only_feature:  ###
                logits_t, _ = self.classifier(mixed_input[num_s:])
                ssl_loss = torch.mean((torch.softmax(logits_t, dim=1) - all_targets[num_s:]) ** 2)
                loss_g = ssl_loss * coeff * self.args.trade_off
                self.optimizer_f.zero_grad()
                self.optimizer_g.zero_grad()
                loss_g.backward()  ## acculate the gradient
                self.optimizer_f.zero_grad()
                # self.optimizer_g.zero_grad()  ## ssl loss only on feature
                logits_s, _ = self.classifier(mixed_input[:num_s])
                cls_loss = -torch.mean(torch.sum(F.log_softmax(logits_s, dim=1) * mixed_target[:num_s], dim=1))
                loss_f = cls_loss
                loss_f.backward()
                self.optimizer_f.step()
                self.optimizer_g.step()
                loss = cls_loss + ssl_loss * self.args.trade_off
            else:
                logits_t, _ = self.classifier(mixed_input[num_s:])
                ssl_loss = torch.mean((torch.softmax(logits_t, dim=1) - all_targets[num_s:]) ** 2)
                loss_g = ssl_loss * coeff * self.args.trade_off
                self.optimizer_f.zero_grad()
                self.optimizer_g.zero_grad()
                loss_g.backward()  ## acculate the gradient

                logits_s, _ = self.classifier(mixed_input[:num_s])
                cls_loss = -torch.mean(torch.sum(F.log_softmax(logits_s, dim=1) * mixed_target[:num_s], dim=1))
                loss_f = cls_loss
                loss_f.backward()
                self.optimizer_f.step()
                self.optimizer_g.step()
                loss = cls_loss + ssl_loss * self.args.trade_off

            prec1_iter = accuracy(logits_s, source_gt)
            prec1.update(prec1_iter, source_data.size(0))

            losses_cls.update(cls_loss.item(), source_data.size(0))
            losses_ssl.update(ssl_loss.item(), source_data.size(0))
            losses_all.update(loss.item(), source_data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            self.iters += 1
            if i % 10 == 0:
                print("  Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LCls:%3f, Lssl:%3f, LossAll:%3f, Acc1_rough:%3f" % \
                      (self.iters, self.epoch, self.args.epochs, data_time.avg, batch_time.avg, losses_cls.avg, losses_ssl.avg, losses_all.avg,  prec1.avg))

        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write("  Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LCls:%3f, Lssl:%3f, LossAll:%3f, Acc1_rough:%3f" % \
                      (self.iters, self.epoch, self.args.epochs, data_time.avg, batch_time.avg, losses_cls.avg, losses_ssl.avg, losses_all.avg,  prec1.avg))
        log.close()

