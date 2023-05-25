#! /bin/bash

export CUDA_VISIBLE_DEVICES=1
python fs_mol/multitask_test.py\
        outputs/Completed/FSMol_Multitask_2023-05-17_02-26-14/best_model.pt\
        datasets/fs-mol/\
        --regression-task\
        --task-list-file datasets/entire_train_set.json\
        --metric-to-use rmse
# python fs_mol/multitask_test.py TRAINED_MODEL DATA_PATH


# LOG for model weights
 # FSMol_Multitask_2023-05-03_07-49-15
 # outputs/FSMol_Multitask_2023-05-13_16-50-08/best_model.pt\
 # outputs/Completed/FSMol_Multitask_2023-05-17_02-26-14   # full train