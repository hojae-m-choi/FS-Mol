#! /bin/bash

export CUDA_VISIBLE_DEVICES=1

python fs_mol/multitask_train.py\
    datasets/fs-mol/\
    --regression-task\
    --task-list-file datasets/entire_train_set.json\
    --metric-to-use rmse
