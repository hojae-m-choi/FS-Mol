#! /usr/bin/bash

export CUDA_VISIBLE_DEVICES=1
export LD_LIBARY_PATH=/anaconda/envs/fsmol/lib:$LD_LIBARY_PATH
python fs_mol/maml_test.py\
            datasets/fs-mol/\
            --trained-model outputs/FSMol_MAML_2023-05-04_07-23-48/best_validation.pkl\
            --regression-task --test-metric rmse