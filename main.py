import torch
import argparse
import os
from torch.backends import cudnn
import random
import json
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import warnings

def opts():
    parser = argparse.ArgumentParser(description='Train script.')
    ### On data loader
    parser.add_argument('--source', type=str, default='Art', help='source domain')
    parser.add_argument('--target', type=str, default='Clipart', help='target domain')
    parser.add_argument('--dataset', type=str, default='DomainNet', help='OfficeHome | office | multi')
    parser.add_argument('--datapath', type=str, default='./data/', help='datapath')
    parser.add_argument('--transform_type', type=str, default='randomcrop', help='randomcrop | randomsizedcrop | center')
    parser.add_argument('--batchsize', type=int, default=32, help='number of labeled examples in the target')
    parser.add_argument('--num_class', type=int, default=126, help='number of classes')
    parser.add_argument('--num_workers', type=int, default=8, help='number of works for data loader')

    parser.add_argument('--category_mean', action='store_true', default=False,
                        help='for visda, if true, the score is the mean over all categories instead of all samples')
    parser.add_argument('--strongaug', action='store_true', default=False,
                        help='whether use the strong augmentation (i.e., RandomAug) it is True in FixMatch and UDA')

    parser.add_argument('--trade_off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    ## specific hyper-parameter_for CDAN
    parser.add_argument('--entropy_cdan', default=False, action='store_true', help='use entropy conditioning')
    ## specific hyper-parameter_for MCD
    parser.add_argument('--num_k', type=int, default=4, help='number of step C in mcd')
    ## specific hyper-parameter_for MDD
    parser.add_argument('--bottleneck_dim', type=int, default=256, help='dimension of the bottleneck')
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")

    ## specific hyper-parameter_for Fixmatch
    parser.add_argument('--use_ema', default=False, action='store_true', help='whether test with ema loss')
    parser.add_argument('--p_cutoff', type=float, default=0.95, help="threshold for the pseudo label")
    parser.add_argument('--ema_m', type=float, default=0.999, help="threshold for the pseudo label")
    parser.add_argument('--mu', type=int, default=1, help='unlabeled batch size / labeled batch size')

    parser.add_argument('--T', type=float, default=1.0, help="temperature of softmax")
    parser.add_argument('--vat_eps', type=float, default=1.0, help="eps for the vat method")
    ## McDalNets hyper-parameter
    parser.add_argument('--mcdalnet_dis', type=str, default='L1', help='which network to use')
    parser.add_argument('--onlyT', action='store_true', default=False, help='')

    ## On distributed
    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:10002', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--port', type=str, default='12355')


    parser.add_argument('--test_only', action='store_true', default=False, help='no training, load checkpoint and conduct test')
    # parser.add_argument('--AlphaGraph', type=float, default=0.75, help='momentum')
    # parser.add_argument('--graphk', type=int, default=20, help='k-nearest graph for the LP algorithm only')

    ### On model construction
    parser.add_argument('--net', type=str, default='resnet50', help='which network to use')

    ### On optimization
    parser.add_argument('--epochs', type=int, default=30, help='maximum number of iterations to train (default: 5)')
    parser.add_argument('--iters_per_epoch', type=int, default=250, help='iters_per_epoch')
    parser.add_argument('--base_lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--lr_schedule', type=str, default='inv', help='lr change schedule')
    parser.add_argument('--regular_only_feature', action='store_true', default=False, help='')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    ### others
    parser.add_argument('--saving', action='store_true', default=False, help='')
    parser.add_argument('--method', help='set the method to use', default='MixMatch', type=str)
    parser.add_argument('--resume', help='set the resume file path', default='', type=str)
    parser.add_argument('--exp_name', help='the log name', default='log', type=str)
    parser.add_argument('--save_dir', help='the log file', default='mixmatch', type=str)
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--calculate_a_dis_only', action='store_true', default=False, help='')


    args = parser.parse_args()

    if args.dataset == 'OfficeHome':
        args.num_class = 65
    elif args.dataset == 'Office31':
        args.num_class = 31
    elif args.dataset == 'DomainNet':  ## for closed set DA.
        args.num_class = 345
    elif args.dataset == 'VisDA2017':
        args.num_class = 12
        args.category_mean = True  ## additionally record the mean accuracy over categories
        args.transform_type = 'center'
        print('!! Adopting the center transform and calculate the mean accuracy over cateogires for the VisDA-2017 dataset')

    # data = datetime.date.today()
    # args.exp_name = str(data.month) + str(data.day) + '_' + args.exp_name
    args.exp_name = args.exp_name + '_' + args.dataset + '_' + args.method + '_' + args.net + '_' + args.source + '2' + args.target

    return args


def main():
    args = opts()
    cudnn.benchmark = True
    print('Called with args:')
    print(args)

    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    print('Output will be saved to %s.' % args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    log = open(os.path.join(args.save_dir, 'log.txt'), 'a')
    log.write("\n")
    log.write('\n------------ training start ------------------\n')
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n-------------------------------------------\n')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batchsize = int(args.batchsize / ngpus_per_node)
        args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print('batch size on each gpu is:', args.batchsize)
    num_classes = args.num_class


    import common.vision.models as models
    backbone = models.__dict__[args.net](pretrained=True)
    pre_trained_feature = True

    from common.modules.classifier import Classifier
    if args.method == 'Fixmatch':
        args.strongaug = True
        classifier = Classifier(backbone, num_classes, pre_trained=pre_trained_feature)
        from solver.solver_fixmatch import Solver as Solver
    elif args.method == 'UDA':
        args.strongaug = True
        classifier = Classifier(backbone, num_classes, pre_trained=pre_trained_feature)
        from solver.solver_uda import Solver as Solver
    elif args.method == 'EntropyMini':
        classifier = Classifier(backbone, num_classes, pre_trained=pre_trained_feature)
        from solver.solver_entropy_minimization import Solver as Solver
    elif args.method == 'PseudoLabel':
        classifier = Classifier(backbone, num_classes, pre_trained=pre_trained_feature)
        from solver.solver_pseudo_label import Solver as Solver
    elif args.method == 'PiModel':
        args.strongaug = False
        classifier = Classifier(backbone, num_classes, pre_trained=pre_trained_feature)
        from solver.solver_pi import Solver as Solver
    elif args.method == 'MeanTeacher':
        classifier = Classifier(backbone, num_classes, pre_trained=pre_trained_feature)
        from solver.solver_mean_teacher import Solver as Solver
    elif args.method == 'VAT':
        classifier = Classifier(backbone, num_classes, pre_trained=pre_trained_feature)
        from solver.solver_vat import Solver as Solver
    elif args.method == 'Mixmatch':
        args.strongaug = False
        classifier = Classifier(backbone, num_classes, pre_trained=pre_trained_feature)
        from solver.solver_mixmatch import Solver as Solver

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            ## syn BN, since we set broadcast_buffers=False, so we should use it to synbn
            classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
            classifier.cuda(args.gpu)
            ## broadcast_buffers=False --> to enable twice forward before a backward !!! e.g., M(S-data), M(T-data), loss.backward().
            classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu], broadcast_buffers=False)
        else:
            raise NotImplementedError
    else:
        classifier.cuda(args.gpu)

    from data.prepare_data_da import generate_dataloader as Dataloader
    dataloaders = Dataloader(args)
    train_solver = Solver(classifier, dataloaders, args)
    train_solver.solve()


if __name__ == '__main__':
    main()



