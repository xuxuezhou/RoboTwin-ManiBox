#!/bin/bash

# ManiBox Data Processor - with GPU memory management
# Usage: ./run_manibox.sh <task_name> <task_config> <num_episodes> [batch_size] [use_cpu]

if [ $# -lt 3 ]; then
    echo "🚀 ManiBox Data Processor"
    echo "Usage: $0 <task_name> <task_config> <num_episodes> [batch_size] [use_cpu]"
    echo ""
    echo "Examples:"
    echo "  $0 grasp_apple default 50                    # Default GPU processing"
    echo "  $0 grasp_apple default 50 2                  # GPU with batch_size=2"
    echo "  $0 grasp_apple default 50 4 cpu              # Force CPU processing"
    echo ""
    echo "GPU Memory Tips:"
    echo "  - If GPU memory error: reduce batch_size (try 2 or 1)"
    echo "  - If still error: use 'cpu' as 5th argument"
    exit 1
fi

TASK_NAME=$1
TASK_CONFIG=$2
NUM_EPISODES=$3
BATCH_SIZE=${4:-4}         # Default batch size 4
USE_CPU=${5:-""}           # Default empty (use GPU)

echo "🚀 ManiBox Data Processor"
echo "================================"
echo "Task: $TASK_NAME"
echo "Config: $TASK_CONFIG" 
echo "Episodes: $NUM_EPISODES"
echo "Batch Size: $BATCH_SIZE"

# Build command
CMD="python process_data.py $TASK_NAME $TASK_CONFIG $NUM_EPISODES --batch_size $BATCH_SIZE"

if [ "$USE_CPU" = "cpu" ]; then
    CMD="$CMD --cpu"
    echo "Mode: CPU (slower but safe)"
else
    echo "Mode: GPU with memory management"
fi

echo "================================"

# Run the command
eval $CMD

if [ $? -eq 0 ]; then
    echo "================================"
    echo "🎉 Processing completed successfully!"
    echo "Output: processed_data/manibox-${TASK_NAME}/integration.pkl"
else
    echo "================================"
    echo "❌ Processing failed!"
    echo ""
    echo "💡 Try these solutions:"
    echo "  1. Reduce batch size: $0 $TASK_NAME $TASK_CONFIG $NUM_EPISODES 2"
    echo "  2. Use CPU mode: $0 $TASK_NAME $TASK_CONFIG $NUM_EPISODES 4 cpu"
    exit 1
fi 