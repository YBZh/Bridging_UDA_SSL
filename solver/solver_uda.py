## we conduct the DA methods following https://github.com/thuml/Transfer-Learning-Library, which gives better results than the common settings.

import torch
import os
import time
from .base_solver import BaseSolver
import torch.nn.functional as F
from utils.utils import AverageMeter, to_cuda, accuracy, accuracy_for_each_class, get_random_recover_index
from dalib.ssl.uda import consistency_loss


class Solver(BaseSolver):
    def __init__(self, classifier, dataloaders, args, **kwargs):
        super(Solver, self).__init__(classifier, dataloaders, args, **kwargs)

        self.total_iters = self.args.epochs * self.args.iters_per_epoch
        self.gamma = 0.001
        self.decay_rate = 0.75
        self.build_optimizer()
        self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
        self.train_data['target']['iterator'] = iter(self.train_data['target']['loader'])
        assert args.strongaug  ## StrongAug is expected.
        assert self.args.mu == 7

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

            (source_data, _), source_gt = self.get_samples('source')
            (target_data_weak, target_data_strong), target_gt_for_visual = self.get_samples('target')
            data_time.update(time.time() - end)

            source_data = source_data.cuda(self.args.gpu, non_blocking=True)
            target_data_weak = target_data_weak.cuda(self.args.gpu, non_blocking=True)
            target_data_strong = target_data_strong.cuda(self.args.gpu, non_blocking=True)
            source_gt = source_gt.cuda(self.args.gpu, non_blocking=True)
            target_gt_for_visual = target_gt_for_visual.cuda(self.args.gpu, non_blocking=True)

            if self.args.regular_only_feature:  ###
                data = torch.cat((target_data_weak, target_data_strong), dim=0)
                logit, _ = self.classifier(data)
                logits_t_w, logits_t_s = logit.chunk(2, dim=0)
                ssl_loss, mask, acc_selected = consistency_loss(logits_t_w, logits_t_s,
                                                            target_gt_for_visual, T=1.0, p_cutoff=self.args.p_cutoff)
                loss_g = ssl_loss * self.args.trade_off
                self.optimizer_f.zero_grad()
                self.optimizer_g.zero_grad()
                loss_g.backward()
                self.optimizer_f.zero_grad()

                logits_s, _ = self.classifier(source_data)
                cls_loss = F.cross_entropy(logits_s, source_gt)
                loss_f = cls_loss
                loss_f.backward()
                self.optimizer_f.step()
                self.optimizer_g.step()
                loss = cls_loss + ssl_loss * self.args.trade_off
            else:
                data = torch.cat((target_data_weak, target_data_strong), dim=0)
                logit, _ = self.classifier(data)
                logits_t_w, logits_t_s = logit.chunk(2, dim=0)
                ssl_loss, mask, acc_selected = consistency_loss(logits_t_w, logits_t_s,
                                                            target_gt_for_visual, T=1.0, p_cutoff=self.args.p_cutoff)
                loss_g = ssl_loss * self.args.trade_off
                self.optimizer_f.zero_grad()
                self.optimizer_g.zero_grad()
                loss_g.backward()  ## accumulate the gradient

                logits_s, _ = self.classifier(source_data)
                cls_loss = F.cross_entropy(logits_s, source_gt)
                loss_f = cls_loss
                loss_f.backward()
                self.optimizer_f.step()
                self.optimizer_g.step()
                loss = cls_loss + ssl_loss * self.args.trade_off

            prec1_iter = accuracy(logits_s, source_gt)
            prec1.update(prec1_iter, source_data.size(0))
            losses_cls.update(cls_loss.item(), source_data.size(0))
            losses_ssl.update(ssl_loss.item(), data.size(0))
            losses_all.update(loss.item(), source_data.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            self.iters += 1
            if i % 10 == 0:
                print("  Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LCls:%3f, Lssl:%3f, LossAll:%3f, Acc1:%3f, SNum: %3f, SAcc: %3f" % \
                      (self.iters, self.epoch, self.args.epochs, data_time.avg, batch_time.avg, losses_cls.avg, losses_ssl.avg, losses_all.avg,  prec1.avg, mask.item(), acc_selected))

        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write("  Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LCls:%3f, Lssl:%3f, LossAll:%3f, Acc1:%3f, SNum: %3f, SAcc: %3f" % \
                      (self.iters, self.epoch, self.args.epochs, data_time.avg, batch_time.avg, losses_cls.avg, losses_ssl.avg, losses_all.avg,  prec1.avg, mask.item(), acc_selected))
        log.close()






