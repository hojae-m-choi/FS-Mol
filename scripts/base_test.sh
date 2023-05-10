#! /usr/bin/bash

export CUDA_VISIBLE_DEVICES=1
python fs_mol/baseline_test.py\
            datasets/fs-mol/\
            --regression-task\
            --grid-search False