# Deploy ACT on RoboTwin
## 1. Environment Setup
The conda environment for ACT with RoboTwin is identical to the official ACT environment. Please follow the ([ACT official documentation](https://tonyzhaozh.github.io/aloha/)) to install the environment and directly overwrite the RoboTwin virtual environment in [INSTALLATION.md](../../INSTALLATION.md).
```bash
pip install pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython

cd adetr && pip install -e .
```
## 2. Generate RoboTwin Data

```bash
# task_name:you can see our current supported task names under `task_config` directory
# gpu_id: The GPU ID for data generation
# This will generate a given number of episodes of data
# episode num for each task is defined in task_config/${task_name}.yml
bash run_task.sh ${task_name} ${gpu_id} 
```
For example
```
bash run_task.sh bottle_adjust 0
```
See [RoboTwin Tutorial (Usage Section)](../../README.md) for more details.

## 3. Generate HDF5 Data
> HDF5 is the data format required for ACT training.

run the following in the `RoboTwin/` root directory:

```bash
# task_name: same as step two's task_name
# head_camera_type: Defaults to D435.
# expert_data_num: The number of episodes of data to be converted to HDF5. In accord with or below the num of episodes you have generated in step two
# embodiment_name: embodiments are defined in `assets/embodiments/${embodiment_name}`
# gpu_id: The GPU ID for data format transfer.
# After running, the data will be saved to `policy/ACT/data` by default.
bash process_data_act.sh $task_name $head_camera_type $expert_data_num $embodiment_name $gpu_id
```
For example
```bash
bash process_data_act.sh bottle_adjust D435 50 aloha-agilex-1 0
```

If success, you will find the `sim_${task_name}_${head_camera_type}_${expert_data_num}` folder under `policy/ACT/data`, with the following data structure:
```
data/
├── sim_${task_name}_${head_camera_type}_${expert_data_num}
│   ├── episode_0.hdf5
│   ├── episode_1.hdf5
│   ├── ...
```

## 4. Train
First add an task config item in `policy/ACT/constants.py`
```python 
SIM_TASK_CONFIGS = {
    ...
    'sim_bottle_adjust_D435_50': {
        'dataset_dir': DATA_DIR + '/sim_bottle_adjust_D435_50',
        'num_episodes': 50, # in accord with the data you have generated in step two
        'episode_len': 500,
        'camera_names': ["cam_high", "cam_right_wrist", "cam_left_wrist"]
    }
}
```
Then begin the training
```bash
bash train.sh $task_name $head_camera_type $expert_data_num $seed $gpu_id
```
For example:
```bash
bash train.sh bottle_adjust D435 50 0 0
```
## 5. Eval on RoboTwin
   
Once the model training is complete, you can test your model's performance on the RoboTwin simulation platform.

```bash
bash eval.sh $task_name $seed $gpu_id
```
If you want to enable temporal aggregation to smooth the arm move, add `--temporal_agg`

For example
```bash
bash eval.sh bottle_adjust 0 0 --temporal_agg
```

The eval script default stops after collecting 100 success trials. You can modify the 
`test_num = 100` in `eval_policy_act.py`
