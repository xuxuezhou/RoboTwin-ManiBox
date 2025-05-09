#!/bin/bash
task_name=${1}
setting=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

DEBUG=False
save_ckpt=True

# addition_info=train
# exp_name=${task_name}-act-${addition_info}
# run_dir="data/outputs/${exp_name}_seed${seed}"

# echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# if [ $DEBUG = True ]; then
#     echo -e "\033[33mDebug mode!\033[0m"
#     wandb_mode=offline
# else
#     echo -e "\033[33mTrain mode\033[0m"
#     wandb_mode=online
# fi

# if [ ! -d "./data/${task_name}_${head_camera_type}_${expert_data_num}" ]; then
#     echo "Processed data not found, running pkl2hdf5.py ..."
#     cd ../..
#     expert_data_num_minus_one=$((expert_data_num - 1))
#     if [ ! -d "./data/${task_name}_${head_camera_type}/episode${expert_data_num_minus_one}" ]; then
#         echo "error: expert data does not exist"
#         exit 1
#     else
#         python script/pkl2hdf5.py ${task_name} ${head_camera_type} ${expert_data_num}
#         cd policy/ACT
#     fi
# fi

export CUDA_VISIBLE_DEVICES=${gpu_id}

python3 imitate_episodes.py \
    --task_name sim-${task_name}-${setting}-${expert_data_num} \
    --ckpt_dir ./checkpoints/act-${task_name}/${setting}-${expert_data_num} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 1e-5 \
    --seed ${seed}
