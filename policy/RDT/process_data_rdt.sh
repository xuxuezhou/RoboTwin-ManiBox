task_name=${1}
setting=${2}
expert_data_num=${3}
gpu_id=${4}

export CUDA_VISIBLE_DEVICES=${gpu_id}
python scripts/pkl2hdf5_rdt.py $task_name $setting $expert_data_num