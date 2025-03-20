#!/bin/bash
export VISION_TACTILE_ON=0

task_name=${1}
gpu_id=${2}

if [ ${task_name} == "classify_tactile" ]; then
    export VISION_TACTILE_ON=1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo ${task_name} | python script/run_task.py