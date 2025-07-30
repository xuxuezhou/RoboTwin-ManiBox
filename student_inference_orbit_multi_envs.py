# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
python source/standalone/workflows/rsl_rl/student_inference_orbit_multi_envs.py --task Isaac-Lift-Cube-MobileAloha-Play-v0  --ckpt_dir /home/orbit/xxz_code/low-level-thk/ckpt/2024-09-16_16-57-31RNN --policy_class RNN --ckpt_name policy_best.ckpt --nheads 20 --seed 0
"""
"""Script to play a checkpoint if an RL agent from RSL-RL."""

from __future__ import annotations
import os
os.environ["WANDB_DISABLED"] = "true"

from pprint import pprint
import json
import time

"""Launch Isaac Sim Simulator first."""
import argparse
import pickle
from omni.isaac.lab.app import AppLauncher
import cli_args  # isort: skip
from VFCNet.train import make_policy, set_model_config
from VFCNet.yolo_process_data import YoloProcessDataByTimeStep

# Simulator play
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--asdasd", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# General inference usage
parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                    default=[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 1.0], required=False)
# parser.add_argument('--initial_qpos', action='store', type=list, help='initial_qpos', 
#                     default=[-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3393220901489258, 
#                              -0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, -0.3397035598754883], 
#                     required=False)

parser.add_argument('--max_pos_lookahead', action='store', type=int, help='max_pos_lookahead', default=0, required=False)
parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
parser.add_argument('--use_accelerate', action='store', type=bool, help='whether use accelerate', default=False, required=False)

# for RNNCNN
parser.add_argument('--rnn_layers', action='store', type=int, help='rnn_layers', default=3, required=False)
parser.add_argument('--rnn_hidden_dim', action='store', type=int, help='rnn_hidden_dim', default=512, required=False)
parser.add_argument('--actor_hidden_dim', action='store', type=int, help='actor_hidden_dim', default=512, required=False)
parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base', default=False, required=False)
parser.add_argument('--arm_delay_time', action='store', type=int, help='arm_delay_time', default=0, required=False)
parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image', default=False, required=False)
parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default='./ckpt')
parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize, CNNMLP, ACT, Diffusion', default='ACT', required=False)
parser.add_argument('--dilation', default=False, action='store_true',help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),help="Type of positional embedding to use on top of the image features", required=False)
parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")
parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
parser.add_argument('--load_config', action='store', type=int, help='load_config', default=1, required=False)
parser.add_argument('--device', type=str, help='device', default='cuda:0')
parser.add_argument('--num_train_step', action='store', type=int, help='num_train_step', default=10000, required=False)
# parser.add_argument('--context_len', action='store', type=int, help='context_len', default=16, required=False)

# for ACT
parser.add_argument('--pre_norm', action='store_true', required=False)
parser.add_argument('--dropout', default=0.2, type=float, help="Dropout applied in the transformer", required=False)
parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=32, required=False)
parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=32, required=False)
parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)

# for Diffusion
parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import traceback
import numpy as np
from einops import rearrange
import carb
import random

from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from omni.isaac.lab_tasks.utils.data_collector.cobot_data_collect import project_to_rotation, transfer_sim_image_to_standard_img_tensor
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from VFCNet.policy.CNNRNNPolicy import CNNRNNPolicy
from VFCNet.policy.RNNPolicy import RNNPolicy

from datetime import datetime
time_now = datetime.now()
timestamp = time_now.strftime("%Y-%m-%d_%H-%M-%S")

import IPython
e = IPython.embed

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if use GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_config(args):
    task_config = {'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}
    args.state_dim = args.state_dim if not args.use_robot_base else args.state_dim + 2  # qpos=7
    args.action_dim = args.state_dim
   
    if args.load_config:
        # load the policy config from the old model as well
        config_file = os.path.join(args.ckpt_dir, 'config.json') # 'logs/rsl_rl/2024-06-07_22-16-34CNNRNN/config.json'
        with open(config_file, 'r') as f:
            config_json = f.read()
        policy_config = json.loads(config_json)['policy_config']
        print(f"The training config {args.ckpt_dir} has been synced for inference!")
    
    # policy_config = set_model_config(args, task_config['camera_names'], 0)
     
    config = {
        'ckpt_dir': args.ckpt_dir,
        'policy_class': args.policy_class,
        'policy_config': policy_config,
        'seed': args.seed,
        'pretrain_ckpt_dir': args.ckpt_dir,
        'use_accelerate': args.use_accelerate,
        'device': args.device,
        'pretrain_ckpt_dir': args.ckpt_dir,
    }
    
    return config


def inverse_project_to_rotation(qpos: torch.Tensor) -> torch.Tensor:
    # Extract the gripper qpos values from the 7th column
    qpos_gripper = qpos[:, 6]
    
    # Calculate the simulated gripper values
    qpos_gripper_sim_value = (qpos_gripper + 0.14160313) / 107.43576967

    # Extract the arm qpos values from the first 6 columns
    qpos_arm = qpos[:, :6]

    # Expand to the last two dimensions to align with the simulator
    qpos_gripper_sim = qpos_gripper_sim_value.unsqueeze(1).expand(-1, 2)

    # Concatenate the arm and gripper qpos values
    qpos_sim = torch.cat((qpos_arm, qpos_gripper_sim), dim=1).to('cuda:0')
    return qpos_sim


def interpolate_action(argsg, prev_action, cur_action, use_interp=True):
    prev_action = prev_action.cpu().numpy()
    cur_action = cur_action.cpu().numpy()
    
    steps = np.concatenate((np.array(argsg.arm_steps_length), np.array(argsg.arm_steps_length)), axis=0) #  default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]
    diff = np.abs(cur_action - prev_action)
    
    step = np.ceil(diff / steps).astype(int)
    max_step = np.max(step)
    if max_step <= 1 or not use_interp:
        return torch.from_numpy(cur_action[np.newaxis, :])
    
    new_actions = np.linspace(prev_action, cur_action, max_step*2)
    new_actions = torch.from_numpy(new_actions[1:])
    
    return new_actions


def obs_format_align(obs, stats, env):
    pre_pos_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

    # Assuming obs is a torch tensor and is already on the GPU
    qpos = obs[:, 0:6]  # left arm
    left_gripper_qpos = obs[:, 6:8]
    
    left_gripper_qpos_mean = torch.mean(left_gripper_qpos, dim=1)
    left_gripper_qpos_mean_value = torch.tensor([project_to_rotation(offset.item()) for offset in left_gripper_qpos_mean], device='cuda')
    left_gripper_qpos_mean_value = left_gripper_qpos_mean_value.view(-1, 1)

    # Concatenate left gripper
    qpos = torch.cat([qpos, left_gripper_qpos_mean_value], dim=-1)
    
    # 直接设置左臂qpos而不是从observation获取
    # qpos = sim_next_qpos[:,:-1] # (1,7)

    # Concatenate right arm and right gripper
    right_zeros = torch.zeros((qpos.shape[0], 7), device='cuda')
    qpos = torch.cat([qpos, right_zeros], dim=-1)

    # Normalize qpos
    preprocessed_qpos = pre_pos_process(qpos).float() # shape=(1,14)

    # concatenate images from three cameras
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    images = []
    
    for cam_name in camera_names:
        cam_data = env.unwrapped.scene.sensors[cam_name].data.output["rgb"]
        processed_image = transfer_sim_image_to_standard_img_tensor(cam_data)
        # (num_envs, 3, 480, 640)
        # processed_image = cam_data.float() / 255.0

        # # drop the alpha channel to convert to RGB
        # processed_image = processed_image[:, :, :, :3]
        # processed_image = rearrange(processed_image, 'b h w c -> b c h w') 
        
        images.append(processed_image)
    stacked_images = torch.stack(images, dim=0).cuda().float()
    # (cam3, num_envs, 3, 480, 640)
        
    return qpos, preprocessed_qpos, stacked_images


def main():
    set_seed(args.seed)
    # Create environment
    env_cfg = parse_env_cfg(args.task, use_gpu=not args.cpu, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.seed(args.seed)
    
    # Load policy
    config = get_model_config(args)
    policy = make_policy(config['policy_class'], config['policy_config'], config['pretrain_ckpt_dir']).to(device=args.device)
    policy.eval()
    # policy.eval()
    
    # Load stats data if needed
    if isinstance(policy, CNNRNNPolicy):
        is_qpos_normalized = True
        yolo_preprocess = False
        stats_path = os.path.join(config['pretrain_ckpt_dir'], args.ckpt_stats_name)
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        stats['qpos_mean'] = torch.tensor(stats['qpos_mean'], device=args.device)
        stats['qpos_std'] = torch.tensor(stats['qpos_std'], device=args.device)
        post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']
    else:
        policy: RNNPolicy
        is_qpos_normalized = False
        yolo_preprocess = True
        stats = {'qpos_mean': 0, 'qpos_std': 1}
        post_process = lambda a: a

    reset_flag = True
    
    if yolo_preprocess:
        yolo_process_data = YoloProcessDataByTimeStep()
    
    success_count = 0
    total_runs = 0
    FAST_EVAL = True
    MAX_EVAL_RUN = args.nheads
    ITR_NUM = 6 if FAST_EVAL else 100
    # height_offset = 0.2968
    # xy_threshold = 0.10
    # z_threshold = 0.042
    height_offset = 0.31
    xy_threshold = 0.08
    z_threshold = 0.05
    start_time = time.time()
    use_interp = False
    
    # Open sim-test.log in append mode
    log_file_path = 'sim-test.log'
    log_file = open(log_file_path, 'a')  # Ensure the file is opened once
    xyz_threshold = 0.05
    try:
        while simulation_app.is_running():
            
            if reset_flag:
                reset_flag = False
                
                # Initialize environment and observations
                env.reset()
                obs, obs_dict = env.get_observations()
                original_qpos, preprocessed_observed_qpos, stacked_images = obs_format_align(obs_dict['observations']['student'], stats, env)
                
                # Initialize variables
                bs = args.num_envs
                device = args.device
                steps = 0
                inner_step = 0
                success = torch.zeros((bs, 1), dtype=torch.bool, device=device)
                terminate = torch.zeros((bs, 1), dtype=torch.bool, device=device)
                initial_object_height = env.unwrapped.scene['object'].data.root_pos_w[:, 2].unsqueeze(1)
                drop_steps = 3
                policy.reset_recur(bs, device)
                if yolo_preprocess:
                    yolo_process_data.reset_new_episode()
                
                
                object_pos = env.unwrapped.scene['object'].data.root_pos_w
                tablecloth_pos = env.unwrapped.scene['platform'].data.root_pos_w 
                object_pos_str = np.array2string(object_pos.cpu().numpy(), precision=4, suppress_small=True, separator=', ')
                tablecloth_pos_str = np.array2string(tablecloth_pos.cpu().numpy(), precision=4, suppress_small=True, separator=', ')

                log_file.write("Object Position:\n")
                log_file.write(object_pos_str + "\n")
                log_file.write("Tablecloth Position:\n")
                log_file.write(tablecloth_pos_str + "\n")
            
            ###########################################################################################
            
            dones = torch.zeros((bs, 1), dtype=torch.bool, device=device)
            # Update initial_object_height and reset variables for done environments
            # if 'dones' in locals() and dones.any():
            #     initial_object_height[dones, 0] = env.unwrapped.scene['object'].data.root_pos_w[dones.squeeze(), 2]
            #     success[dones] = False
            #     steps = 0
            #     inner_step = 0

            # Prepare the mask
            # mask = ((inner_step < ITR_NUM) & (~terminate) & (steps >= 3))
            # # mask = torch.zeros_like(mask, dtype=torch.bool)s
            # mask = mask.to(device)

            # Prepare robot state
            robot_state = preprocessed_observed_qpos if is_qpos_normalized else original_qpos
            robot_state = robot_state.float()
            # print(f"robot state is {robot_state}")
            # print(f"Current steps is {steps} with drop mask {drop_mask}")

            # Process images
            if yolo_preprocess:
                # stacked_images: (cam3, num_envs, 3, 480, 640)
                processed_images = yolo_process_data.parallel_process_traj(
                    stacked_images[0],  
                    stacked_images[1], 
                    stacked_images[2]   
                ).to(device=args.device)
                # bboxes: (num_env, 12)
            else:
                processed_images = stacked_images
            image = processed_images.float()

            with torch.inference_mode():
                # Get policy output
                original_next_qpos = policy(image, None, robot_state, None, None, None, None)
                zero_action = torch.zeros_like(original_next_qpos.to(device=env.unwrapped.device))
                # if steps < drop_steps, use zero action
                if steps < drop_steps:
                    original_next_qpos = zero_action
                # original_next_qpos = torch.where(steps < drop_steps, zero_action, original_next_qpos)
            # original_next_qpos = torch.where(drop_mask, original_next_qpos, robot_state)
            postprocessed_next_qpos = post_process(original_next_qpos)

            # Interpolate actions
            interp_qpos = interpolate_action(args, original_qpos, postprocessed_next_qpos, use_interp=use_interp)
            # if =True: [17, num_envs, 14]
            # # Prepare action
            # action = torch.where(mask, original_qpos, interp_qpos[-1].to(device=original_qpos.device))
            env.env.episode_length_buf += 1
            env.env.common_step_counter += 1
            
            # while mask.any():
            loop_limit = interp_qpos.shape[0] if use_interp else ITR_NUM
            for inner_step in range(loop_limit):
                # mask: =0: done, =1: not done, continue
                # only execute those env where mask is 1
                
                # repeat the action for the same number of steps
                
                # Compute sim_next_qpos
                if use_interp:
                    sim_next_qpos = inverse_project_to_rotation(interp_qpos[inner_step, :]) 
                else:
                    sim_next_qpos = inverse_project_to_rotation(interp_qpos[-1]) 
                    
                # interp_qpos.pop(0)
                
                if FAST_EVAL:
                    sim_next_qvel = torch.zeros_like(sim_next_qpos)
                    env.unwrapped.scene['robot'].write_joint_state_to_sim(sim_next_qpos, sim_next_qvel)
                else:
                    env.unwrapped.scene['robot'].set_joint_position_target(sim_next_qpos)

                # Step the environment
                obs, rew, dones, extras = env.step_fake()
                term_cfg = env.env.reward_manager.get_term_cfg('reaching_object')
                reaching_rew = term_cfg.func(env.env, **term_cfg.params)
                import  omni.isaac.lab_tasks.manager_based.manipulation.lift.mdp as mdp
                from omni.isaac.lab.managers import SceneEntityCfg
                quat_rew = mdp.object_quat_similar(env.env, 0.1, SceneEntityCfg("object"), SceneEntityCfg("ee_frame"))
                # print("rew", rew.item(), "reaching", reaching_rew.item(), "quat", quat_rew.item())
                # # success:() rew > 1), reaching > 1.0, quat > 1.4
                success_in_this_step = torch.logical_and(
                    reaching_rew > 0.6,
                    quat_rew > 1.4
                ).unsqueeze(1)  # (num_envs, 1)
                
                # Update object height and check for success
                object_height = env.unwrapped.scene['object'].data.root_pos_w[:, 2].unsqueeze(1)
                height_diff = object_height - initial_object_height
                ee_w = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :]
                cube_pos_w = env.unwrapped.scene["object"].data.root_pos_w 
                # platform_height = env.unwrapped.scene['platform'].data.root_pos_w[:, 2].unsqueeze(1) + height_offset
                # print("platform_height", platform_height, initial_object_height - 0.035)
                platform_height = initial_object_height - 0.04  # 0.035 is apple radium
                distance = torch.norm(cube_pos_w - ee_w, dim=1).unsqueeze(1)
                
                position_diff = torch.abs(ee_w - cube_pos_w)
                # print(position_diff)
                # hand_bottom_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :]
                # hand_top_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 1, :]
                # hand_middle_pos = (hand_bottom_pos + hand_top_pos) / 2
                # position_diff = torch.abs(hand_middle_pos - cube_pos_w)
                # xyz_below_threshold = (torch.all(position_diff[..., :3] < xyz_threshold, dim=1)).unsqueeze(1)
                # print("position_diff[..., :3] max", position_diff[..., :3].max(dim=1).values.item())
                # success_in_this_step = xyz_below_threshold
                # xy_below_threshold = (torch.all(position_diff[..., :2] < xy_threshold, dim=1)).unsqueeze(1)
                # z_below_threshold = (position_diff[..., 2] < z_threshold).unsqueeze(1)
                
                # arm_too_low = ee_w[:, 2] - initial_object_height
                # 一旦成功就可以退出，一旦发现明显失败就可以提前退出。
                # success_in_this_step = torch.logical_or(
                #                     height_diff >= 0.03,
                #                     torch.logical_and(
                #                         torch.logical_and(xy_below_threshold, z_below_threshold), 
                #                         torch.logical_and(22 <= steps, steps <= 24)
                #                     )
                #                 )
                
                # fail_mask could be used as early stopping and reject the success
                fail_in_this_step = (steps > 30) \
                            | (ee_w[:, 2].unsqueeze(1) < platform_height) \
                            # | ((steps >= 24) & (distance > 0.12))
                # attention of platform_height: # too low, will crash
                
                # if failed, set the success_in_this_step to False
                success_in_this_step = torch.logical_and(success_in_this_step, ~fail_in_this_step)
                
                success |= success_in_this_step
                terminate: torch.Tensor
                terminate |= success_in_this_step
                terminate |= fail_in_this_step
                inner_step += 1
                # fail_mask | success_mask
                
                # mask = ((i < ITR_NUM) & (~terminate) & (steps >= 3))
                # if use_interp:
                #     mask = ((inner_step < interp_qpos.shape[0]) & (~terminate) & (steps >= 3))
                # else:
                #     mask = ((inner_step < ITR_NUM) & (~terminate) & (steps >= 3))
                if terminate.all():
                    break
                
                # 1 1 0 0 1
                # TODO: mask仅用来表示内循环是否结束
                # 因为我用success来记录一轮中的各个环境是否成功，即使某个环境done了也没关系，后续随便怎么推理，不影响success的结果（只要曾经是1过）
                    
            # if not mask.any():
            #     # mask.shape: torch.Size([num_envs, 1])
            #     # NOTE: update iterator
            #     steps += 1
            #     inner_step = 0
            # import pdb; pdb.set_trace()
            # print("rew", rew)
            steps += 1
            if 22 <= steps <= 25:
                print("reaching_rew", reaching_rew, "quat_rew", quat_rew)
            # print("steps", steps, "success", success.sum().item(), "terminate", terminate.sum().item())
                
            # Get new observations
            obs, obs_dict = env.get_observations()
            original_qpos, preprocessed_observed_qpos, stacked_images = obs_format_align(obs_dict['observations']['student'], stats, env)

            # Check if all environments are done
            if terminate.all() or (terminate.sum().item() + total_runs >= MAX_EVAL_RUN):
                for idx in torch.nonzero(terminate, as_tuple=True)[0]:
                    total_runs += 1
                    if success[idx]:
                        success_count += 1
                        success_message = f"Trajectory {total_runs} successfully lifted the apple. Current success rate: {success_count / total_runs * 100:.2f}%"
                        print(success_message)
                        # Write success to log
                        log_file.write(f"{datetime.now()}: Trajectory {total_runs} SUCCESS - Current Success Rate: {success_count / total_runs * 100:.2f}%\n")
                    else:
                        failure_message = f"Trajectory {total_runs} failed to lift the apple. Current success rate: {success_count / total_runs * 100:.2f}%"
                        print(failure_message)
                        # Write failure to log
                        log_file.write(f"{datetime.now()}: Trajectory {total_runs} FAILURE - Current Success Rate: {success_count / total_runs * 100:.2f}%\n")
                if total_runs >= MAX_EVAL_RUN:
                    break
                    
                # reset all environment
                print("All environments done. Resetting.")
                reset_flag = True
                log_file.flush()

    finally:
        # Close the log file
        log_file.close()

    # Close the simulator
    env.close()

    # Print summary
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    success_rate = (success_count / total_runs) * 100 if total_runs > 0 else 0
    print(f"Total eval time is {elapsed_time:.2f} minutes")
    print(f"Final success rate: {success_rate:.2f}%")
    with open('sim-test.log', 'a') as log_file:
        log_file.write(f"{datetime.now()}: seed = {args.seed}, num_traj = {args.num_train_step}, Success Rate = {success_rate:.2f}%, Elapsed Time = {elapsed_time:.2f} minutes\n")

if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
