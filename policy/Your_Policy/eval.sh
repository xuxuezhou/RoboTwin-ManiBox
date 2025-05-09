#!/bin/bash

policy_name=Your_Policy # modify this
task_name=${1}
seed=${2}
gpu_id=${3}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --seed ${seed} \
    --policy_name ${policy_name}
