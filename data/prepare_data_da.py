import os
import random
import torch
from torchvision import transforms
# import torchvision.datasets as datasets
from data.randaugment import RandAugment
import common.vision.datasets as datasets
from common.vision.transforms import ResizeImage, MultipleApply

def _select_image_process(DATA_TRANSFORM_TYPE):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if DATA_TRANSFORM_TYPE == 'randomcrop':
        transforms_train_weak = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_train_strong = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                RandAugment(2, 10),
                transforms.ToTensor(),
                normalize,
            ])
        transforms_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    elif DATA_TRANSFORM_TYPE == 'randomsizedcrop':   ## default image pro-process in DALIB: https://github.com/thuml/Transfer-Learning-Library
        transforms_train_weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transforms_train_strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(2, 10),
            transforms.ToTensor(),
            normalize
        ])
        transforms_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    elif DATA_TRANSFORM_TYPE == 'center':   ## Only apply to VisDA-2017 dataset following DALIB
        transforms_train_weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transforms_train_strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(2, 10),
            transforms.ToTensor(),
            normalize
        ])
        transforms_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    else:
        raise NotImplementedError

    return transforms_train_weak, transforms_train_strong, transforms_test

def generate_dataloader(args):
    dataloaders = {}
    base_path = os.path.join(args.datapath, args.dataset)
    dataset = datasets.__dict__[args.dataset]
    transforms_train_weak, transforms_train_strong, transforms_test = _select_image_process(args.transform_type)
    if not args.strongaug: ## Sample two same batch with weak augmentation.
        transforms_train_strong = transforms_train_weak

    source_dataset = dataset(root=base_path, task=args.source, download=True, transform=MultipleApply([transforms_train_weak, transforms_train_strong]))
    target_dataset = dataset(root=base_path, task=args.target, download=True, transform=MultipleApply([transforms_train_weak, transforms_train_strong]))
    trans_test_dataset = dataset(root=base_path, task=args.target, download=True, transform=transforms_test)
    # source_dataset = ImageFolder_Strong(file_source, transforms_train_weak, transforms_train_strong, path=True)
    # target_dataset = ImageFolder_Strong(file_target, transforms_train_weak, transforms_train_strong, path=True)

    # trans_test_dataset = datasets.ImageFolder(file_target, transforms_test)
    if args.multiprocessing_distributed:    ## prepare data for distributed multigpu.
        source_sampler = torch.utils.data.distributed.DistributedSampler(source_dataset)
        target_sampler = torch.utils.data.distributed.DistributedSampler(target_dataset)
        # if file_target != file_test:
        if args.dataset == 'DomainNet':
            induc_test_dataset = dataset(root=base_path, task=args.target, split='test', download=True, transform=transforms_test)
            ## large batch size for fast test
            target_loader_induc_test = torch.utils.data.DataLoader(induc_test_dataset,
                                            batch_size=args.batchsize * 5, num_workers=args.num_workers, shuffle=False, drop_last=False)
            dataloaders['induc_test'] = target_loader_induc_test
        elif args.dataset == 'VisDA2017':
            induc_test_dataset = dataset(root=base_path, task="test", download=True, transform=transforms_test)
            ## large batch size for fast test
            target_loader_induc_test = torch.utils.data.DataLoader(induc_test_dataset,
                                            batch_size=args.batchsize * 5, num_workers=args.num_workers, shuffle=False, drop_last=False)
            dataloaders['induc_test'] = target_loader_induc_test


        source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batchsize,
                                                    num_workers=args.num_workers, sampler=source_sampler, drop_last=True)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=int(args.batchsize * args.mu),
                                                    num_workers=args.num_workers, sampler=target_sampler, drop_last=True)
        target_loader_trans_test = torch.utils.data.DataLoader(trans_test_dataset,
                                                               batch_size=args.batchsize * 5, num_workers=args.num_workers, shuffle=False,
                                                               drop_last=False)   ## large batch size for fast test
        dataloaders['s_sampler'] = source_sampler
        dataloaders['t_sampler'] = target_sampler
    else:
        if args.dataset == 'DomainNet':
            induc_test_dataset = dataset(root=base_path, task=args.target, split='test', download=True, transform=transforms_test)
            ## large batch size for fast test
            target_loader_induc_test = torch.utils.data.DataLoader(induc_test_dataset,
                                            batch_size=args.batchsize * 5, num_workers=args.num_workers, shuffle=False, drop_last=False)
            dataloaders['induc_test'] = target_loader_induc_test
        elif args.dataset == 'VisDA2017':
            induc_test_dataset = dataset(root=base_path, task="test", download=True, transform=transforms_test)
            target_loader_induc_test = torch.utils.data.DataLoader(induc_test_dataset,
                                            batch_size=args.batchsize * 5, num_workers=args.num_workers, shuffle=False, drop_last=False)
            dataloaders['induc_test'] = target_loader_induc_test
        source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batchsize,
                                                    num_workers=args.num_workers, shuffle=True, drop_last=True)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=int(args.batchsize * args.mu),
                                        num_workers=args.num_workers, shuffle=True, drop_last=True)
        target_loader_trans_test = torch.utils.data.DataLoader(trans_test_dataset,
                                        batch_size=args.batchsize * 5, num_workers=args.num_workers, shuffle=False, drop_last=False)

    dataloaders['source'] = source_loader
    dataloaders['target'] = target_loader
    dataloaders['trans_test'] = target_loader_trans_test

    return dataloaders
