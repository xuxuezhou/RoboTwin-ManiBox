# bash script/run_eval_policy.sh mug_hanging 0

DEBUG=False

task_name=${1}
gpu_id=${2}

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python script/eval_policy.py ${task_name}

                            
