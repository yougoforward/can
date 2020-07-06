# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_can --dataset pascal_aug \
    --model new_can3 --aux --atrous-rates 12 24 36 --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname new_can3_res50_pascal_aug

#test [single-scale]
python -m experiments.segmentation.test_whole --dataset pascal_aug \
    --model new_can3 --aux --atrous-rates 12 24 36 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pascal_aug/new_can3/new_can3_res50_pascal_aug/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test_whole --dataset pascal_aug \
    --model new_can3 --aux --atrous-rates 12 24 36 --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/pascal_aug/new_can3/new_can3_res50_pascal_aug/checkpoint.pth.tar --split val --mode testval --ms