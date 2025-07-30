#!/bin/bash

# SimpleBBoxDiffusionPolicy训练脚本
# 使用简化的BBox-based Diffusion policy

echo "=== 开始SimpleBBoxDiffusionPolicy训练 ==="

# 设置matplotlib为非交互模式
export MPLBACKEND=Agg

# SimpleBBoxDiffusionPolicy训练参数
python manibox/ManiBox/train.py \
    --policy_class SimpleBBoxDiffusion \
    --batch_size 32 \
    --dataset /home/xuxuezhou/code/RoboTwin/policy/ManiBox/processed_data/manibox-pick-diverse-bottles \
    --num_episodes 500 \
    --loss_function l1 \
    --num_epochs 200 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --gradient_accumulation_steps 1 \
    --num_objects 2 \
    --action_horizon 8 \
    --observation_horizon 1 \
    --num_inference_timesteps 20 \
    --warmup_ratio 0.1 \
    --scheduler cos \
    --num_objects 2 \
    --objects 'bottle, bottle' \
    > simple_bbox_diffusion_training.log 2>&1

echo "=== SimpleBBoxDiffusionPolicy训练完成 ==="
echo "检查日志文件: simple_bbox_diffusion_training.log"
echo "检查checkpoint目录: ckpt/" 