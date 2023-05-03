#! /bin/bash

export CUDA_VISIBLE_DEVICES=1
python fs_mol/multitask_train.py datasets/fs-mol/ --regression-task --metric-to-use rmse
