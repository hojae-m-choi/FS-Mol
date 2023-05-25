#! /usr/bin/bash

export CUDA_VISIBLE_DEVICES=1
python fs_mol/baseline_test.py\
            datasets/fs-mol/\
            --regression-task\
            --grid-search False\
            --train-sizes "[375]"\
            --test-size 30\
            --active-learning\
            --heuristic var_ensemble
            
            
            
            # random