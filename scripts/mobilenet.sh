#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#python3 -m torch.distributed.launch --nproc_per_node=8 mobilenet_prune.py -a resnet50 --pretrained --start-epoch 0 --lr 0.128 --batch-size 128 --workers 8 --opt-level O1 --sync_bn /data/ImageNet_new &&
python3 -m torch.distributed.launch --nproc_per_node=8 mobilenet_prune.py -a resnet50 --resume checkpoint.pth.tar --start-epoch 81 --lr 10 --batch-size 128 --prune 0.50 --regularization --record --workers 8 --opt-level O1 --sync_bn /data/ImageNet_new &&
python3 -m torch.distributed.launch --nproc_per_node=8 mobilenet_prune.py -a resnet50 --resume checkpoint.pth.tar --start-epoch 52 --freeze --lr 0.128 --batch-size 128 --prune 0.50 --workers 8 --opt-level O1 --sync_bn /data/ImageNet_new 