#! /bin/bash

export CUDA_VISIBLE_DEVICES=1
python fs_mol/multitask_test.py\
        outputs/FSMol_Multitask_2023-05-03_07-49-15/best_model.pt\
        datasets/fs-mol/\
        --regression-task\
        --metric-to-use rmse
# python fs_mol/multitask_test.py TRAINED_MODEL DATA_PATH