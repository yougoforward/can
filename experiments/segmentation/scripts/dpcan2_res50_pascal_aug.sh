# !/usr/bin/env bash
# train
python -m experiments.segmentation.train_can --dataset psacal_aug \
    --model dpcan2 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --checkname dpcan2_res50_psacal_aug

#test [single-scale]
python -m experiments.segmentation.test --dataset psacal_aug \
    --model dpcan2 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/psacal_aug/dpcan2/dpcan2_res50_psacal_aug/checkpoint.pth.tar --split val --mode testval

#test [multi-scale]
python -m experiments.segmentation.test --dataset psacal_aug \
    --model dpcan2 --aux --dilated --base-size 520 --crop-size 520 \
    --backbone resnet50 --resume experiments/segmentation/runs/psacal_aug/dpcan2/dpcan2_res50_psacal_aug/checkpoint.pth.tar --split val --mode testval --ms