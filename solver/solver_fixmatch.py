## we conduct the DA methods following https://github.com/thuml/Transfer-Learning-Library, which gives better results than the common settings.

import torch
import os
import torch.nn as nn
import time
from .base_solver import BaseSolver
import torch.nn.functional as F
from utils.utils import AverageMeter, to_cuda, accuracy, accuracy_for_each_class, get_random_recover_index
from tools.lr_scheduler import StepwiseLR, InvLR
import copy


class Solver(BaseSolver):
    def __init__(self, classifier, dataloaders, args, **kwargs):
        super(Solver, self).__init__(classifier, dataloaders, args, **kwargs)

        self.classifier_ema = copy.deepcopy(self.classifier)
        self.total_iters = self.args.epochs * self.args.iters_per_epoch
        self.ema_m = args.ema_m
        self.gamma = 0.001
        self.decay_rate = 0.75
        self.build_optimizer()
        self.train_data['source']['iterator'] = iter(self.train_data['source']['loader'])
        self.train_data['target']['iterator'] = iter(self.train_data['target']['loader'])
        # define loss function (criterion) and optimizer
        self.Cri_CE = nn.CrossEntropyLoss().cuda(args.gpu)
        self.Cri_CE_noreduce = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)
        assert args.strongaug ## StrongAug is expected.
        assert self.args.mu == 7
        print('solver has been initialized')
    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(self.classifier.parameters(), self.classifier_ema.parameters()):
            param_eval.copy_(param_eval * self.args.ema_m + param_train.detach() * (1 - self.args.ema_m))
        for buffer_train, buffer_eval in zip(self.classifier.buffers(), self.classifier_ema.buffers()):
            buffer_eval.copy_(buffer_train)


    def update_network(self, **kwargs):
        self.classifier.train()
        # self.F1.train()
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

            ## why only supoort distribute one model
            # print(self.classifier.module.backbone.conv1.weight[0][0][0][:3])
            # print(self.classifier.module.head.bias[:5])

            (source_data, _), source_gt = self.get_samples('source')
            (target_data_weak, target_data_strong), target_gt_for_visual  = self.get_samples('target')
            data_time.update(time.time() - end)
            ##
            source_data = source_data.cuda(self.args.gpu, non_blocking=True)
            target_data_weak = target_data_weak.cuda(self.args.gpu, non_blocking=True)
            target_data_strong = target_data_strong.cuda(self.args.gpu, non_blocking=True)
            source_gt = source_gt.cuda(self.args.gpu, non_blocking=True)
            target_gt_for_visual = target_gt_for_visual.cuda(self.args.gpu, non_blocking=True)

            # source_num = source_data.size(0)
            # data = torch.cat((source_data, target_data_weak, target_data_strong), dim=0)
            # logit_all, _ = self.classifier(data)
            # logits_s = logit_all[:source_num, :]
            # logits_t = logit_all[source_num:, :]
            # logits_t_w, logits_t_s = logits_t.chunk(2, dim=0)



            ### pytorch example, conducting two backward should be implemented with the following way.
            # >> > ddp = torch.nn.parallel.DistributedDataParallel(model, pg)
            # >> > with ddp.no_sync():
            #     >> > for input in inputs:
            #         >> > ddp(input).backward()  # no synchronization, accumulate grads
            # >> > ddp(another_input).backward()  # synchronize grads

            if self.args.regular_only_feature:  ###
                data = torch.cat((target_data_weak, target_data_strong), dim=0)
                logit, _ = self.classifier(data)
                logits_t_w, logits_t_s = logit.chunk(2, dim=0)
                ssl_loss, mask, acc_selected = self.consistency_loss(logits_t_w,
                                                                     logits_t_s, target_gt_for_visual,
                                                                     'ce', 1.0, self.args.p_cutoff,
                                                                     use_hard_labels=True)
                loss_g = ssl_loss * self.args.trade_off
                self.optimizer_f.zero_grad()
                self.optimizer_g.zero_grad()
                # with self.classifier.no_sync():
                loss_g.backward()
                # print("\n 1 ", self.classifier.module.backbone.conv1.weight.grad[0][0][0][:3])
                # print("\n 1 ", self.classifier.module.head.bias.grad[:5])
                # temp_grad = []
                # for param in self.classifier.module.backbone.parameters():
                #     temp_grad.append(param.grad.clone())
                self.optimizer_f.zero_grad()
                # self.optimizer_g.zero_grad()
                # print("\n 2 ", self.classifier.module.backbone.conv1.weight.grad[0][0][0][:3])
                # print("\n 2 ", self.classifier.module.head.bias.grad[:5])
                logits_s, _ = self.classifier(source_data)
                cls_loss = F.cross_entropy(logits_s, source_gt)
                loss_f = cls_loss
                loss_f.backward()
                # print("\n 3 ", self.classifier.module.backbone.conv1.weight.grad[0][0][0][:3])
                # print("\n 3 ", self.classifier.module.head.bias.grad[:5])
                # raise NotImplementedError
                # count = 0
                # for param in self.classifier.module.backbone.parameters():
                #     param.grad.zero_()
                #     param.grad = temp_grad[count]
                #     count = count + 1
                self.optimizer_f.step()
                self.optimizer_g.step()
                loss = cls_loss + ssl_loss * self.args.trade_off
            else:
                data = torch.cat((target_data_weak, target_data_strong), dim=0)
                logit, _ = self.classifier(data)
                logits_t_w, logits_t_s = logit.chunk(2, dim=0)
                ssl_loss, mask, acc_selected = self.consistency_loss(logits_t_w,
                                                                     logits_t_s, target_gt_for_visual,
                                                                     'ce', 1.0, self.args.p_cutoff,
                                                                     use_hard_labels=True)
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
            self.optimizer_f.zero_grad()
            self.optimizer_g.zero_grad()
            if self.args.use_ema:
                with torch.no_grad():
                    self._eval_model_update()

            losses_cls.update(cls_loss.item(), source_data.size(0))
            losses_ssl.update(ssl_loss.item(), data.size(0))
            losses_all.update(loss.item(), source_data.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            self.iters += 1
            if i % 20 == 0:
                print("  Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LCls:%3f, Lssl:%3f, LossAll:%3f, Acc1:%3f, SNum: %3f, SAcc: %3f" % \
                      (self.iters, self.epoch, self.args.epochs, data_time.avg, batch_time.avg, losses_cls.avg, losses_ssl.avg, losses_all.avg,  prec1.avg, mask.item(), acc_selected))

        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write("  Train:epoch: %d:[%d/%d], Tdata: %3f, Tbatch: %3f, LCls:%3f, Lssl:%3f, LossAll:%3f, Acc1:%3f, SNum: %3f, SAcc: %3f" % \
                      (self.iters, self.epoch, self.args.epochs, data_time.avg, batch_time.avg, losses_cls.avg, losses_ssl.avg, losses_all.avg,  prec1.avg, mask.item(), acc_selected))
        log.close()

    def build_optimizer(self):
        self.optimizer_g = torch.optim.SGD([
            {'params': self.classifier.module.backbone.parameters(), "lr_mult": self.lr_mul_feat},
            {'params': self.classifier.module.bottleneck.parameters(), "lr_mult": 1.0} ## the bottleneck is just a identitical map, parameter-free
        ], momentum=0.9, lr=self.args.base_lr, weight_decay=self.args.wd, nesterov=True)
        self.optimizer_f = torch.optim.SGD([{'params': self.classifier.module.head.parameters(), "lr_mult": 1.0}], momentum=0.9, lr=self.args.base_lr,
                                           weight_decay=self.args.wd, nesterov=True)
        ## the learning rate schedule in the original SymNets paper.
        if self.args.lr_schedule == 'dalib':
            self.lr_sheduler_g = StepwiseLR(self.optimizer_g, init_lr=self.args.base_lr, gamma=self.gamma,
                                            decay_rate=self.decay_rate)
            self.lr_sheduler_f = StepwiseLR(self.optimizer_f, init_lr=self.args.base_lr, gamma=self.gamma,
                                            decay_rate=self.decay_rate)
        elif self.args.lr_schedule == 'inv':
            self.lr_sheduler_g = InvLR(self.optimizer_g, init_lr=self.args.base_lr, total_iters=self.total_iters)
            self.lr_sheduler_f = InvLR(self.optimizer_f, init_lr=self.args.base_lr, total_iters=self.total_iters)
        else:
            raise NotImplementedError

    def consistency_loss(self, logits_w, logits_s, target_gt_for_visual, name='ce', T=1.0, p_cutoff=0.0,
                         use_hard_labels=True):
        assert name in ['ce', 'L2']
        logits_w = logits_w.detach()
        if name == 'L2':
            raise NotImplementedError
            # assert logits_w.size() == logits_s.size()
            # return F.mse_loss(logits_s, logits_w, reduction='mean')
        elif name == 'ce':
            pseudo_label = torch.softmax(logits_w, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            # print('max score is: %3f, mean score is: %3f' % (max(max_probs).item(), max_probs.mean().item()))
            mask_binary = max_probs.ge(p_cutoff)
            mask = mask_binary.float()

            if mask.mean().item() == 0:
                acc_selected = 0
            else:
                acc_selected = (target_gt_for_visual[mask_binary] == max_idx[mask_binary]).float().mean().item()

            if use_hard_labels:
                masked_loss = self.Cri_CE_noreduce(logits_s, max_idx) * mask
            else:
                raise NotImplementedError
                # pseudo_label = torch.softmax(logits_w / T, dim=-1)
                # masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
            return masked_loss.mean(), mask.mean(), acc_selected

        else:
            assert Exception('Not Implemented consistency_loss')

    def test(self):
        print('begin testing')
        if self.args.use_ema:
            eval_classifier = self.classifier_ema
        else:
            eval_classifier =  self.classifier
        eval_classifier.eval()
        # self.classifier.eval()
        prec1 = AverageMeter()
        if self.args.category_mean:
            counter_all_ft = torch.FloatTensor(self.args.num_class).fill_(0)
            counter_acc_ft = torch.FloatTensor(self.args.num_class).fill_(0)

        for i, (input, target) in enumerate(self.test_data['loader']):
            input, target = to_cuda(input), to_cuda(target)
            with torch.no_grad():
                output_test, _ = eval_classifier(input)
            prec1_iter = accuracy(output_test, target)
            prec1.update(prec1_iter, input.size(0))
            if self.args.category_mean:
                counter_all_ft, counter_acc_ft = accuracy_for_each_class(output_test, target,
                                                                         counter_all_ft, counter_acc_ft)

        print("                       TracnTest:epoch: %d, iter: %d, MeanSampleAcc: %3f" % \
              (self.epoch, self.iters, prec1.avg))


        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write(
            "                                                                                                                                                                                                                                TracnTest:epoch: %d, iter: %d, MeanSampleAcc: %3f" % \
            (self.epoch, self.iters, prec1.avg))
        log.close()
        is_best = prec1.avg > self.best_prec1
        self.best_prec1 = max(self.best_prec1, prec1.avg)

        if self.args.category_mean:
            acc_for_each_class_ft = counter_acc_ft / counter_all_ft
            log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
            log.write("\nClass-wise Acc of Ft:")  ## based on the task classifier.
            for i in range(self.args.num_class):
                if i == 0:
                    log.write("%dst: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                else:
                    log.write(", %dth: %3f" % (i + 1, acc_for_each_class_ft[i]))
            log.write("\n")
            log.write(
                "                                                                                                                                                                                                                             TracnTest:epoch: %d, iter: %d, MeanCateAcc: %3f" % \
                (self.epoch, self.iters, acc_for_each_class_ft.mean()))
            log.close()
            is_best = acc_for_each_class_ft.mean() > self.best_prec1_cate
            self.best_prec1_cate = max(self.best_prec1_cate, acc_for_each_class_ft.mean())
            return is_best

        else:
            return is_best

    def induc_test(self):
        print('begin inductesting')
        # self.classifier.eval()
        if self.args.use_ema:
            eval_classifier = self.classifier_ema
        else:
            eval_classifier =  self.classifier
        eval_classifier.eval()
        prec1 = AverageMeter()
        if self.args.category_mean:
            counter_all_ft = torch.FloatTensor(self.args.num_class).fill_(0)
            counter_acc_ft = torch.FloatTensor(self.args.num_class).fill_(0)

        for i, (input, target) in enumerate(self.train_data['induc_test']['loader']):
            input, target = to_cuda(input), to_cuda(target)
            with torch.no_grad():
                output_test, _ = eval_classifier(input)
            prec1_iter = accuracy(output_test, target)
            prec1.update(prec1_iter, input.size(0))
            if self.args.category_mean:
                counter_all_ft, counter_acc_ft = accuracy_for_each_class(output_test, target,
                                                                         counter_all_ft, counter_acc_ft)

        print("                       InducTest:epoch: %d, iter: %d, MeanSampleAcc: %3f" % \
              (self.epoch, self.iters, prec1.avg))

        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write(
            "                                                                                                                                                                                                                        InducTest:epoch: %d, iter: %d, MeanSampleAcc: %3f" % \
            (self.epoch, self.iters, prec1.avg))
        log.close()
        is_best = prec1.avg > self.best_prec1_induc
        self.best_prec1_induc = max(self.best_prec1_induc, prec1.avg)
        if self.args.category_mean:
            acc_for_each_class_ft = counter_acc_ft / counter_all_ft
            log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
            log.write("\nClass-wise Acc of Ft:")  ## based on the task classifier.
            for i in range(self.args.num_class):
                if i == 0:
                    log.write("%dst: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 1:
                    log.write(",  %dnd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                elif i == 2:
                    log.write(", %drd: %3f" % (i + 1, acc_for_each_class_ft[i]))
                else:
                    log.write(", %dth: %3f" % (i + 1, acc_for_each_class_ft[i]))
            log.write("\n")
            log.write(
                "                                                                                                                                                                                                                   InducTest:epoch: %d, iter: %d, MeanCateAcc: %3f" % \
                (self.epoch, self.iters, acc_for_each_class_ft.mean()))
            log.close()
            is_best = acc_for_each_class_ft.mean() > self.best_prec1_induc_cate
            self.best_prec1_induc_cate = max(self.best_prec1_induc_cate , acc_for_each_class_ft.mean())
            return is_best
        else:
            return is_best

    # def calculate_A_dis(self):
    #     if self.args.use_ema:
    #         eval_classifier = self.classifier_ema
    #     else:
    #         eval_classifier = self.classifier
    #     ################## prepare source feature and target features. ###############################################
    #     target_u_feature_list = []
    #     print('prepare feature of target unlabeled data')
    #     for i, (input, _) in enumerate(self.test_data['loader']):
    #         input = to_cuda(input)
    #         with torch.no_grad():
    #             logit, target_u_feature_iter = eval_classifier(input)
    #         target_u_feature_list.append(target_u_feature_iter)
    #     target_u_feature_matrix = torch.cat(target_u_feature_list, dim=0)
    #
    #     source_feature_list = []
    #     print('prepare feature of target unlabeled data')
    #     for i, ((input, _), _) in enumerate(self.train_data['source']['loader']):
    #         input = to_cuda(input)
    #         with torch.no_grad():
    #             logit, source_feature_iter = eval_classifier(input)
    #         source_feature_list.append(source_feature_iter)
    #     source_feature_matrix = torch.cat(source_feature_list, dim=0)
    #
    #     # if self.inductive_flag:
    #     #     target_u_feature_list_induc = []
    #     #     for i, (input, target) in enumerate(self.train_data['induc_test']['loader']):
    #     #         input = to_cuda(input)
    #     #         with torch.no_grad():
    #     #             logit, target_u_feature_iter = eval_classifier(input)
    #     #         target_u_feature_list_induc.append(target_u_feature_iter)
    #     #     target_u_feature_matrix_induc = torch.cat(target_u_feature_list_induc, dim=0)
    #     #
    #     #     a_dis_s_induc_t = self.proxy_a_distance(source_feature_matrix, target_u_feature_matrix_induc)
    #     #     a_dis_tran_t_induc_t = self.proxy_a_distance(target_u_feature_matrix, target_u_feature_matrix_induc)
    #     #     log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
    #     #     log.write("\n")
    #     #     log.write('A-distance S_inducT: %3f, transT_inducT: %3f' % (a_dis_s_induc_t, a_dis_tran_t_induc_t))
    #     #     log.close()
    #     a_dis_st = self.proxy_a_distance(source_feature_matrix, target_u_feature_matrix)
    #     log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
    #     log.write("\n")
    #     log.write('A-distance S_transT: %3f' % (a_dis_st))
    #     log.close()