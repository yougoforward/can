# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset psacal_aug \
    --model deeplabv3 --aux --dilated --multi-grid --stride 16 --atrous-rates 6 12 18 --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname deeplabv3_res50_psacal_aug

#test [single-scale]
python -m experiments.segmentation.test --dataset psacal_aug \
    --model deeplabv3 --aux --dilated --multi-grid --stride 16 --atrous-rates 6 12 18 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/psacal_aug/deeplabv3/deeplabv3_res50_psacal_aug/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset psacal_aug \
    --model deeplabv3 --aux --dilated --multi-grid --stride 16 --atrous-rates 6 12 18 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/psacal_aug/deeplabv3/deeplabv3_res50_psacal_aug/checkpoint.pth.tar --split val --mode testval --ms