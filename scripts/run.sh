#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#python3 -m torch.distributed.launch --nproc_per_node=8 main.py -a resnet50 --pretrained --admm --start-epoch 61 --lr 0.4 --b 128 --workers 8 --opt-level O1 --sync_bn /data/ImageNet_new &&
#python3 -m torch.distributed.launch --nproc_per_node=8 main.py -a resnet50 --resume checkpoint.pth.tar --masked_retrain --start-epoch 51 --lr 0.1 --b 128 --workers 8 --opt-level O1 --sync_bn /data/ImageNet_new
python3 -m torch.distributed.launch --nproc_per_node=8 main.py -a resnet50 --pretrained --masked_retrain --start-epoch 42 --lr 0.1 --b 128 --workers 8 --opt-level O1 --sync_bn /data/ImageNet_new