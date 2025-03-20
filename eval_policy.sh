export VISION_TACTILE_ON=0

DEBUG=False

policy_name=${1}
task_name=${2}
ckpt_folder=${3}
gpu_id=${4}
head_camera_type="D435"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ${task_name} == "classify_tactile" ]; then
    export VISION_TACTILE_ON=1
fi

python ./script/eval_policy.py "$task_name" "$head_camera_type" "$policy_name" "$ckpt_folder"   