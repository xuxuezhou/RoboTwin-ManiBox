import sys
import numpy as np
import torch
import os
import pickle
import cv2
import time  # Add import for timestamp
import h5py  # Add import for HDF5
from datetime import datetime  # Add import for datetime formatting
from .act_policy import ACTPolicy
import copy
from argparse import Namespace

timeStatic=0

def encode_obs(observation):
    global timeStatic
    timeStatic += 1
    # Replace the provided observation with data from HDF5
    # observation = inject_train_obs(observation)
    head_cam = observation['observation']['head_camera']['rgb']
    left_cam = observation['observation']['left_camera']['rgb']
    right_cam = observation['observation']['right_camera']['rgb']
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0
    left_cam = np.moveaxis(left_cam, -1, 0) / 255.0
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0
    qpos = observation['joint_action']['left_arm'] + [observation['joint_action']['left_gripper']] + \
           observation['joint_action']['right_arm'] + [observation['joint_action']['right_gripper']]
    return {
        'head_cam': head_cam,
        'left_cam': left_cam,
        'right_cam': right_cam,
        'qpos': qpos
    }

class ACT:
    def __init__(self, args_override=None, RoboTwin_Config=None):
        if args_override is None:
            args_override = {
                'kl_weight': 0.1,  # Default value, can be overridden
                'device': 'cuda:0'
            }
        self.policy = ACTPolicy(args_override, RoboTwin_Config)
        self.device = torch.device(args_override['device'])
        self.policy.to(self.device)
        self.policy.eval()
        
        # Temporal aggregation settings
        self.temporal_agg = args_override.get('temporal_agg', False)
        self.num_queries = args_override['chunk_size']
        self.state_dim = 14  # Standard joint dimension for bimanual robot
        self.max_timesteps = 1000  # Large enough for deployment
        
        # Set query frequency based on temporal_agg - matching imitate_episodes.py logic
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            self.query_frequency = 1
            # Initialize with zeros matching imitate_episodes.py format
            self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps + self.num_queries, self.state_dim]).to(self.device)
            print(f"Temporal aggregation enabled with {self.num_queries} queries")
        
        self.t = 0  # Current timestep
        
        # Load statistics for normalization
        ckpt_dir = args_override.get('ckpt_dir', '')
        if ckpt_dir:
            # Load dataset stats for normalization
            stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    self.stats = pickle.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                print(f"Warning: Could not find stats file at {stats_path}")
                self.stats = None
                
            # Load policy weights
            ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
            print("current pwd:", os.getcwd())
            if os.path.exists(ckpt_path):
                loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
                print(f"Loaded policy weights from {ckpt_path}")
                print(f"Loading status: {loading_status}")
            else:
                print(f"Warning: Could not find policy checkpoint at {ckpt_path}")
        else:
            self.stats = None
        
    def pre_process(self, qpos):
        """Normalize input joint positions"""
        if self.stats is not None:
            return (qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        return qpos

    def post_process(self, action):
        """Denormalize model outputs"""
        if self.stats is not None:
            return action * self.stats['action_std'] + self.stats['action_mean']
        return action

    def get_action(self, obs=None):
        if obs is None:
            return None
            
        # Convert observations to tensors and normalize qpos - matching imitate_episodes.py
        qpos_numpy = np.array(obs['qpos'])
        qpos_normalized = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)
        
        # Prepare images following imitate_episodes.py pattern
        # Stack images from all cameras
        curr_images = []
        camera_names = ['head_cam', 'left_cam', 'right_cam']
        for cam_name in camera_names:
            curr_images.append(obs[cam_name])
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Only query the policy at specified intervals - exactly like imitate_episodes.py
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)
                
            if self.temporal_agg:
                # Match temporal aggregation exactly from imitate_episodes.py
                self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = self.all_actions
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                
                # Use same weighting factor as in imitate_episodes.py
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
                
                raw_action = (
                    actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                # Direct action selection, same as imitate_episodes.py
                raw_action = self.all_actions[:, self.t % self.query_frequency]
            # print(f"agg: {self.temporal_agg} all_actions {self.all_actions.shape} raw_action {raw_action.shape}")
        # Denormalize action
        raw_action = raw_action.cpu().numpy()
        action = self.post_process(raw_action)
        
        self.t += 1
        return action

def get_model(usr_args):
    print("get_model in deploy_policy", usr_args)
    return ACT(usr_args, Namespace(**usr_args))

def eval(TASK_ENV, model, observation):
    '''
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment 
    '''
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()
    
    # Get action from model
    actions = model.get_action(obs)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation

def reset_model(model):
    # Reset temporal aggregation state if enabled
    if model.temporal_agg:
        model.all_time_actions = torch.zeros([model.max_timesteps, model.max_timesteps + model.num_queries, model.state_dim]).to(model.device)
        model.t = 0
        print("Reset temporal aggregation state")
    else:
        model.t = 0