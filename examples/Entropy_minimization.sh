#!/bin/sh
##############
## 1. the --dist_url should be different if multiple programs are conducted on one machine. 1: tcp://127.0.0.1:10013  2: tcp://127.0.0.1:10014
## 2. We conduct three runs for each task with seeds of 1, 2, and 3; only command with seed 1 is given for simplicity.
## 3. You could change the number of used GPUs according to the GPU memory size.
#############
## DomainNet,
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --multiprocessing_distributed --dist_url tcp://127.0.0.1:10013 --trade_off 0.1  --regular_only_feature --seed 1  --epochs 30  --batchsize 128  --iters_per_epoch 500 --wd 0.0005  --source c --target i  --method EntropyMini --save_dir ../log/DomainNet --dataset DomainNet
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --multiprocessing_distributed --dist_url tcp://127.0.0.1:10013 --trade_off 0.1  --regular_only_feature --seed 1  --epochs 30  --batchsize 128  --iters_per_epoch 500 --wd 0.0005  --source i --target p  --method EntropyMini --save_dir ../log/DomainNet --dataset DomainNet
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --multiprocessing_distributed --dist_url tcp://127.0.0.1:10013 --trade_off 0.1  --regular_only_feature --seed 1  --epochs 30  --batchsize 128  --iters_per_epoch 500 --wd 0.0005  --source p --target q  --method EntropyMini --save_dir ../log/DomainNet --dataset DomainNet
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --multiprocessing_distributed --dist_url tcp://127.0.0.1:10013 --trade_off 0.1  --regular_only_feature --seed 1  --epochs 30  --batchsize 128  --iters_per_epoch 500 --wd 0.0005  --source q --target r  --method EntropyMini --save_dir ../log/DomainNet --dataset DomainNet
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --multiprocessing_distributed --dist_url tcp://127.0.0.1:10013 --trade_off 0.1  --regular_only_feature --seed 1  --epochs 30  --batchsize 128  --iters_per_epoch 500 --wd 0.0005  --source r --target s  --method EntropyMini --save_dir ../log/DomainNet --dataset DomainNet
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --multiprocessing_distributed --dist_url tcp://127.0.0.1:10013 --trade_off 0.1  --regular_only_feature --seed 1  --epochs 30  --batchsize 128  --iters_per_epoch 500 --wd 0.0005  --source s --target c  --method EntropyMini --save_dir ../log/DomainNet --dataset DomainNet

## VisDA2017
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../main.py --multiprocessing_distributed --dist_url tcp://127.0.0.1:10013  --regular_only_feature --seed 1 --net resnet101 --transform_type center --category_mean --epochs 30  --batchsize 128  --iters_per_epoch 500 --wd 0.0005  --source Synthetic --target Real --method EntropyMini --save_dir ../log/VisDA2017 --dataset VisDA2017

## OfficeHome
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source Ar --target Cl  --method EntropyMini --save_dir ../log/OfficeHome --dataset OfficeHome
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source Ar --target Rw  --method EntropyMini --save_dir ../log/OfficeHome --dataset OfficeHome
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source Cl --target Pr  --method EntropyMini --save_dir ../log/OfficeHome --dataset OfficeHome
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source Pr --target Ar  --method EntropyMini --save_dir ../log/OfficeHome --dataset OfficeHome
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source Pr --target Rw  --method EntropyMini --save_dir ../log/OfficeHome --dataset OfficeHome
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source Rw --target Cl  --method EntropyMini --save_dir ../log/OfficeHome --dataset OfficeHome

## Office31
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source A --target D  --method EntropyMini --save_dir ../log/Office31 --dataset Office31
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source A --target W  --method EntropyMini --save_dir ../log/Office31 --dataset Office31
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source W --target A  --method EntropyMini --save_dir ../log/Office31 --dataset Office31
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source D --target A  --method EntropyMini --save_dir ../log/Office31 --dataset Office31
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source D --target W  --method EntropyMini --save_dir ../log/Office31 --dataset Office31
CUDA_VISIBLE_DEVICES=0,1 python ../main.py --dist_url tcp://127.0.0.1:10013 --multiprocessing_distributed --regular_only_feature --seed 1  --epochs 30  --batchsize 128 --iters_per_epoch 250 --wd 0.0005  --source W --target D  --method EntropyMini --save_dir ../log/Office31 --dataset Office31

