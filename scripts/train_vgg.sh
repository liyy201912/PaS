#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=8 train_vgg.py -a RepVGG-A0 --start-epoch 0 --epochs 120 --b 128 --workers 8 --opt-level O0 --sync_bn /data/ImageNet_new 