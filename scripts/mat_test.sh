#! /usr/bin/bash

export CUDA_VISIBLE_DEVICES=1
python fs_mol/mat_test.py\
    third_party/MAT/model_weight/pretrained_weights.pt\
    datasets/fs-mol/\
    --regression-task\
    --metric-to-use rmse