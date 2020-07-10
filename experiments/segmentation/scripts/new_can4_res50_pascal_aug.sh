# !/usr/bin/env bash
# train
python -m experiments.segmentation.train --dataset pascal_aug \
    --model new_can4 --aux --atrous-rates 12 24 36--base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname new_can4_res50_pascal_aug

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pascal_aug \
    --model new_can4 --aux --dilated --atrous-rates 12 24 36 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pascal_aug/new_can4/new_can4_res50_pascal_aug/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pascal_aug \
    --model new_can4 --aux --atrous-rates 12 24 36 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pascal_aug/new_can4/new_can4_res50_pascal_aug/checkpoint.pth.tar --split val --mode testval --ms