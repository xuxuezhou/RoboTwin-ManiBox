# bash eval.sh mug_hanging 10 3000 0 0

task_name=${1}
expert_data_num=${2}
checkpoint_num=${3}
seed=${4}
gpu_id=${5}

cd ../..
bash script/run_eval_policy_dp3.sh robot_dp3 $task_name $expert_data_num eval $seed $checkpoint_num $gpu_id
