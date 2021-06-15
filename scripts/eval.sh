#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python3 -m torch.distributed.launch --nproc_per_node=1 autoprune.py -a RepVGG-B0 --resume B1_29G_748_923.pth.tar --start-epoch 81 --lr 0.1 --batch-size 128 --evaluate --workers 8 --opt-level O0 --sync_bn /data/ImageNet_new &&
