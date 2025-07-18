#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

# PYTHONWARNINGS=ignore::UserWarning \
python -u script/collect_data.py $task_name $task_config