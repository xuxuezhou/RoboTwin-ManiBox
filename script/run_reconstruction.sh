#!/bin/bash

# Run data reconstruction with bounding box visualization
# This script reconstructs robot demonstrations using ground truth actions

echo "🚀 Starting data reconstruction with bounding box visualization..."

# Set variables
TASK_NAME="pick_diverse_bottles"
DATA_PATH="policy/ManiBox/processed_data/manibox-pick-diverse-bottles"
EPISODE_ID=0
SEED=0
MAX_STEPS=50  # Limit to first 50 steps for testing
OUTPUT_DIR="./reconstruction_output"

# Activate conda environment
echo "🔧 Activating RoboTwin environment..."
conda activate RoboTwin

# Run reconstruction
echo "📊 Running reconstruction..."
python script/reconstruct_data_with_bbox.py \
    --config policy/ManiBox/reconstruct_config.yml \
    --overrides \
    --task_name $TASK_NAME \
    --episode_id $EPISODE_ID \
    --seed $SEED \
    --max_steps $MAX_STEPS \
    --output_dir $OUTPUT_DIR \
    --data_path $DATA_PATH

echo "✅ Reconstruction completed!"
echo "📁 Videos saved in: $OUTPUT_DIR"
echo "🎬 Check the following files:"
echo "   - episode_${EPISODE_ID}_cam_high_bbox.mp4"
echo "   - episode_${EPISODE_ID}_cam_left_wrist_bbox.mp4"
echo "   - episode_${EPISODE_ID}_cam_right_wrist_bbox.mp4" 