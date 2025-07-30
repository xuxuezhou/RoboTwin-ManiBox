#!/bin/bash

# RNNPolicy训练脚本
# 使用RNN policy进行训练

echo "=== 开始RNNPolicy训练 ==="

# 设置matplotlib为非交互模式
export MPLBACKEND=Agg

# RNNPolicy训练参数
python manibox/ManiBox/train.py \
    --policy_class RNN \
    --batch_size 96 \
    --dataset policy/ManiBox/processed_data/manibox-pick-diverse-bottles \
    --num_episodes 500 \
    --loss_function l1 \
    --rnn_layers 5 \
    --rnn_hidden_dim 1024 \
    --actor_hidden_dim 1024 \
    --num_epochs 100 \
    --lr 1e-3 \
    --gradient_accumulation_steps 1 \
    --num_objects 2 \
    --objects 'bottle, bottle' \
    > rnn_training.log 2>&1

echo "=== RNNPolicy训练完成 ==="
echo "检查日志文件: rnn_training.log"
echo "检查checkpoint目录: ckpt/" 