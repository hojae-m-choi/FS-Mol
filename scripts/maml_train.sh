#! /usr/bin/bash

export CUDA_VISIBLE_DEVICES=1
export LD_LIBARY_PATH=/anaconda/envs/fsmol/lib:$LD_LIBARY_PATH
python fs_mol/maml_train.py datasets/fs-mol/ --regression-task --test-metric rmse