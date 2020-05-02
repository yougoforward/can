#!/usr/bin/env bash
#train
python train_can.py --dataset pcontext \
    --model can --aux --dilated --multi-grid --stride 16 --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname can_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model can --aux --dilated --multi-grid --stride 16 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/can/can_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model can --aux --dilated --multi-grid --stride 16 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/can/can_res50_pcontext/model_best.pth.tar --split val --mode testval --ms