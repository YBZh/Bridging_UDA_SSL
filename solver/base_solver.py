import torch
import torch.nn as nn
import os
from utils.utils import AverageMeter, to_cuda, accuracy, accuracy_for_each_class
from sklearn import svm
import numpy as np
import json
from tools.lr_scheduler import StepwiseLR, InvLR
import shutil

class BaseSolver:
    def __init__(self, classifier, dataloaders, args, **kwargs):
        self.args = args
        self.classifier = classifier
        self.dataloaders = dataloaders
        self.CELoss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.CELoss.cuda(args.gpu)
        self.num_classes = args.num_class

        self.epoch = 0
        self.iters = 0
        self.best_prec1 = 0
        self.best_prec1_induc = 0

        self.best_prec1_cate = 0
        self.best_prec1_induc_cate = 0

        self.iters_per_epoch = args.iters_per_epoch
        self.total_iters = self.args.epochs * self.args.iters_per_epoch
        self.inductive_flag = False
        self.lr_mul_feat = 0.1
        if self.args.net == 'resnet34':
            self.inc = 512
        elif self.args.net == "alexnet":
            self.inc = 4096
        elif args.net == "wideresnet_fixmatch":
            self.inc = 4096
        elif args.net == "wideresnet":
            self.inc = 4096
        elif args.net == "digits":
            self.lr_mul_feat = 1.0

        self.init_data(self.dataloaders)
        self.source_epoch = 0
        self.target_epoch = 0

    def init_data(self, dataloaders):
        self.train_data = {key: dict() for key in dataloaders if key != 'trans_test'}
        for key in self.train_data.keys():
            if key not in dataloaders:
                continue
            cur_dataloader = dataloaders[key]
            self.train_data[key]['loader'] = cur_dataloader
            self.train_data[key]['iterator'] = None

        if 'trans_test' in dataloaders:
            self.test_data = dict()
            self.test_data['loader'] = dataloaders['trans_test']
        if 'induc_test' in dataloaders:
            self.inductive_flag = True

    def solve(self):
        if self.args.calculate_a_dis_only:
            assert self.args.world_size == 1  ## only supporting calculate the A_distance with on GPU
            filename = 'model_last.pth.tar'
            dir_save_file = os.path.join(self.args.save_dir, filename)
            print('load the pre-trained checkpoint from:', dir_save_file)
            checkpoint = torch.load(dir_save_file, map_location='cuda:0')
            self.classifier.load_state_dict(checkpoint)
            self.calculate_A_dis()
            return

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            self.update_network()

            if (epoch % 1 == 0) or (epoch == self.args.epochs -1):
                is_best_trans = self.test()
                if self.inductive_flag:
                    is_best_induc = self.induc_test()
                    if self.args.use_ema:
                        eval_classifier = self.classifier_ema
                    else:
                        eval_classifier = self.classifier
                    self.save_checkpoint(eval_classifier.state_dict(), is_best_induc)
                else:
                    if self.args.use_ema:
                        eval_classifier = self.classifier_ema
                    else:
                        eval_classifier = self.classifier
                    self.save_checkpoint(eval_classifier.state_dict(), is_best_trans)

        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write(" \n \n  Trans Ins :%3f, Trans cate: %3f, Induc Ins :%3f, Induc cate: %3f" % \
                  (self.best_prec1, self.best_prec1_cate, self.best_prec1_induc, self.best_prec1_induc_cate))
        log.close()

        log = open(os.path.join(self.args.save_dir, 'best.txt'), 'a')
        state = {k: v for k, v in self.args._get_kwargs()}
        log.write(json.dumps(state) + '\n')
        log.write(" \n  Trans Ins :%3f, Trans cate: %3f, Induc Ins :%3f, Induc cate: %3f" % \
                  (self.best_prec1, self.best_prec1_cate, self.best_prec1_induc, self.best_prec1_induc_cate))
        log.close()
        # self.calculate_A_dis()

    def save_checkpoint(self, state, is_best=False):
        if self.args.rank == 0:
            filename = 'model_last.pth.tar'
            dir_save_file = os.path.join(self.args.save_dir, filename)
            torch.save(state, dir_save_file)
            if is_best:
                shutil.copyfile(dir_save_file, os.path.join(self.args.save_dir, 'model_best.pth.tar'))

    def get_samples(self, data_name):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader']
        data_iterator = self.train_data[data_name]['iterator']
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            if self.args.multiprocessing_distributed:
                if data_name == 'source':
                    self.source_epoch += 1  ## otherwise, the data order of each epoch is identical.
                    self.train_data['s_sampler']['loader'].set_epoch(self.source_epoch)
                    print('update the random seed of source sample to', self.source_epoch)
                elif data_name == 'target':
                    self.target_epoch += 1  ## otherwise, the data order of each epoch is identical.
                    self.train_data['t_sampler']['loader'].set_epoch(self.target_epoch)
                    print('update the random seed of target sample to', self.target_epoch)
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'] = data_iterator
        return sample


    def update_network(self, **kwargs):
        pass

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

    def test(self):
        print('begin testing')
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
        eval_classifier = self.classifier
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

    def calculate_A_dis(self):
        self.classifier.eval()
        ################## prepare source feature and target features. ###############################################
        target_u_feature_list = []
        print('prepare feature of target unlabeled data')
        for i, (input, _) in enumerate(self.test_data['loader']):
            input = to_cuda(input)
            with torch.no_grad():
                _, target_u_feature_iter = self.classifier(input)
            target_u_feature_list.append(target_u_feature_iter)
        target_u_feature_matrix = torch.cat(target_u_feature_list, dim=0)

        source_feature_list = []
        print('prepare feature of target unlabeled data')
        for i, ((input, _), _) in enumerate(self.train_data['source']['loader']):
            input = to_cuda(input)
            with torch.no_grad():
                _, source_feature_iter = self.classifier(input)
            source_feature_list.append(source_feature_iter)
        source_feature_matrix = torch.cat(source_feature_list, dim=0)

        # if self.inductive_flag:
        #     target_u_feature_list_induc = []
        #     for i, (input, target) in enumerate(self.train_data['induc_test']['loader']):
        #         input = to_cuda(input)
        #         with torch.no_grad():
        #             target_u_feature_iter = self.classifier(input)
        #         target_u_feature_list_induc.append(target_u_feature_iter)
        #     target_u_feature_matrix_induc = torch.cat(target_u_feature_list_induc, dim=0)
        #
        #     a_dis_s_induc_t = self.proxy_a_distance(source_feature_matrix, target_u_feature_matrix_induc)
        #     a_dis_tran_t_induc_t = self.proxy_a_distance(target_u_feature_matrix, target_u_feature_matrix_induc)
        #     log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        #     log.write("\n")
        #     log.write('A-distance S_inducT: %3f, transT_inducT: %3f' % (a_dis_s_induc_t, a_dis_tran_t_induc_t))
        #     log.close()
        a_dis_st = self.proxy_a_distance(source_feature_matrix, target_u_feature_matrix)
        log = open(os.path.join(self.args.save_dir, 'log.txt'), 'a')
        log.write("\n")
        log.write('A-distance S_transT: %3f' % (a_dis_st))
        log.close()

    ##
    def proxy_a_distance(self, source_feature_matrix, target_u_feature_matrix, verbose=True):
        """
        Compute the Proxy-A-Distance of a source/target representation
        """
        num_for_dis = 5000    ##  for datasets of large size, we randomly select a subset to calculate dis for fast speed.
        num_source = source_feature_matrix.size(0)
        num_target = target_u_feature_matrix.size(0)
        if num_source > num_for_dis:
            indices_source = torch.randperm(num_source)
            source_feature_matrix_sampled = source_feature_matrix[indices_source][:num_for_dis]
        else:
            source_feature_matrix_sampled = source_feature_matrix
        if num_target > num_for_dis:
            indices_target = torch.randperm(num_target)
            target_u_feature_matrix_sampled = target_u_feature_matrix[indices_target][:num_for_dis]
        else:
            target_u_feature_matrix_sampled = target_u_feature_matrix
        # a_dis_st = self.proxy_a_distance(source_feature_matrix_sampled.cpu().numpy(),
        #                             target_u_feature_matrix_sampled.cpu().numpy(),
        #                             verbose=True)

        source_X = source_feature_matrix_sampled.cpu().numpy()
        target_X = target_u_feature_matrix_sampled.cpu().numpy()
        nb_source = np.shape(source_X)[0]
        nb_target = np.shape(target_X)[0]
        if verbose:
            print('PAD on', (nb_source, nb_target), 'examples')

        C_list = np.logspace(-5, -1, 5)
        half_source, half_target = int(nb_source / 2), int(nb_target / 2)
        train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
        train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))
        test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
        test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

        best_risk = 1.0
        for C in C_list:
            clf = svm.SVC(C=C, kernel='linear', verbose=False)
            clf.fit(train_X, train_Y)
            train_risk = np.mean(clf.predict(train_X) != train_Y)
            test_risk = np.mean(clf.predict(test_X) != test_Y)
            if verbose:
                print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))
            if test_risk > .5:
                test_risk = 1. - test_risk
            best_risk = min(best_risk, test_risk)
        print('A-distance is:', 2 * (1. - 2 * best_risk))
        return 2 * (1. - 2 * best_risk)
