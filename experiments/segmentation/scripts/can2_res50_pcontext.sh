#!/usr/bin/env bash
#train
python train_can2.py --dataset pcontext \
    --model can2 --aux --dilated --multi-grid --stride 16 --atrous-rates 6 12 18 --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname can2_res50_pcontext

#test [single-scale]
python test.py --dataset pcontext \
    --model can2 --aux --dilated --multi-grid --stride 16 --atrous-rates 6 12 18 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/can2/can2_res50_pcontext/model_best.pth.tar --split val --mode testval

#test [multi-scale]
python test.py --dataset pcontext \
    --model can2 --aux --dilated --multi-grid --stride 16 --atrous-rates 6 12 18 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume runs/pcontext/can2/can2_res50_pcontext/model_best.pth.tar --split val --mode testval --ms