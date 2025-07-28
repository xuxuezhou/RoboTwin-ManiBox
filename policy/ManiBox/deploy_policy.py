#!/usr/bin/env python3
"""
ManiBox Policy Deployment
Deploy ManiBox RNN policy for evaluation with video recording
"""

import sys
import os
import torch
import numpy as np
import json
import yaml
from collections import deque

# Add multiple possible paths for robust importing
current_dir = os.path.dirname(__file__)
manibox_dir = os.path.join(current_dir, "manibox")
manibox_manibox_dir = os.path.join(current_dir, "manibox/ManiBox")
manibox_policy_dir = os.path.join(current_dir, "manibox/ManiBox/policy")

for path in [manibox_dir, manibox_manibox_dir, manibox_policy_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Also ensure we can import from the current working directory
import os
cwd = os.getcwd()
if "RoboTwin" in cwd and "policy/ManiBox" in cwd:
    # We're in ManiBox directory, add the manibox paths
    robotwin_root = cwd.replace("/policy/ManiBox", "")
    manibox_policy_dir = os.path.join(robotwin_root, "policy/ManiBox/manibox")
    if manibox_policy_dir not in sys.path:
        sys.path.insert(0, manibox_policy_dir)

try:
    from yolo_process_data import YoloProcessDataByTimeStep
    from train import make_policy
except ImportError:
    try:
        from ManiBox.yolo_process_data import YoloProcessDataByTimeStep
        from ManiBox.train import make_policy
    except ImportError:
        # Last resort: try absolute import from manibox directory
        manibox_abs_dir = os.path.abspath(os.path.join(current_dir, "manibox"))
        if manibox_abs_dir not in sys.path:
            sys.path.insert(0, manibox_abs_dir)
        from ManiBox.yolo_process_data import YoloProcessDataByTimeStep
        from ManiBox.train import make_policy


class ManiBoxRNNModel:
    """ManiBox RNN model for deployment in evaluation environment"""
    
    def __init__(self, usr_args):
        """Initialize the ManiBox RNN model"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        self.config = usr_args
        self.task_name = usr_args.get('task_name', 'grasp_apple')
        self.ckpt_setting = usr_args.get('ckpt_setting', 'policy_best')
        
        # Construct checkpoint path
        self.ckpt_dir = self._find_latest_checkpoint()
        
        print(f"🤖 Loading ManiBox RNN model from: {self.ckpt_dir}")
        
        # Initialize YOLO processor first
        objects_names = usr_args.get('objects', ['apple'])
        print(f"🔧 Debug: usr_args['objects'] = {usr_args.get('objects')}")
        print(f"🔧 Debug: objects_names = {objects_names}, type = {type(objects_names)}")
        self.yolo_processor = YoloProcessDataByTimeStep(objects_names=objects_names)
        
        # Create policy configuration
        self.policy_config = self._create_policy_config(usr_args, objects_names)
        
        # Create and load the actual RNN model
        self.model = make_policy('RNN', self.policy_config, self.ckpt_dir)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Real ManiBox RNN model loaded successfully!")
        
        # Load statistics for normalization
        self.stats = self._load_stats()
        
        # Initialize observation cache
        self.obs_cache = deque(maxlen=1)
        
        print(f"🎯 ManiBox RNN model initialization complete")
        print(f"   Device: {self.device}")
        print(f"   Objects: {objects_names}")
        print(f"   Task: {self.task_name}")
        print(f"   Expected bbox dim: {len(objects_names) * 3 * 4}")
    
    def _create_policy_config(self, usr_args, objects_names):
        """Create policy configuration for RNN model"""
        # Default camera setup
        camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        
        policy_config = {
            'lr': 2e-3,
            'lr_backbone': 2e-3,
            'epochs': 50,
            'train_loader_len': 4,  # placeholder
            'warmup_ratio': 0.1,
            'use_scheduler': 'cos',
            'backbone': 'resnet18',
            'masks': False,
            'weight_decay': 0.0001,
            'dilation': False,
            'position_embedding': 'sine',
            'loss_function': 'l1',
            'chunk_size': 1,
            'camera_names': camera_names,
            'num_next_action': 0,
            'use_depth_image': False,
            'use_robot_base': False,
            'hidden_dim': 512,
            'device': self.device,
            'state_dim': 14,
            'action_dim': 14,
            'rnn_layers': 3,
            'rnn_hidden_dim': 512,
            'actor_hidden_dim': 512,
            'policy_class': 'RNN',
            'gradient_accumulation_steps': 1
        }
        
        return policy_config
    
    def _load_stats(self):
        """Load dataset statistics for normalization"""
        stats_path = os.path.join(self.ckpt_dir, "dataset_stats.pkl")
        
        if os.path.exists(stats_path):
            try:
                stats = torch.load(stats_path, map_location=self.device)
                print(f"✅ Dataset statistics loaded from {stats_path}")
                return stats
            except Exception as e:
                print(f"⚠️  Failed to load stats file ({e}), computing from processed data...")
        else:
            print(f"⚠️  Stats file not found, computing from processed data...")
        
        return self._compute_stats_from_data()

    def _compute_stats_from_data(self):
        """Compute statistics from processed data files"""
        # Look for processed data files in different possible locations
        possible_data_dirs = [
            f"processed_data/manibox-{self.task_name}",  # 在ManiBox目录下
            f"policy/ManiBox/processed_data/manibox-{self.task_name}",  # 在根目录下
            "processed_data/manibox-beat_block_hammer",  # 在ManiBox目录下
            "policy/ManiBox/processed_data/manibox-beat_block_hammer",  # 在根目录下
            "processed_data/manibox-grasp_apple",  # 在ManiBox目录下
            "policy/ManiBox/processed_data/manibox-grasp_apple",  # 在根目录下
            # 相对于脚本文件的路径
            os.path.join(os.path.dirname(__file__), f"processed_data/manibox-{self.task_name}"),
            os.path.join(os.path.dirname(__file__), "processed_data/manibox-beat_block_hammer"),
            os.path.join(os.path.dirname(__file__), "processed_data/manibox-grasp_apple"),
        ]
        
        for data_dir in possible_data_dirs:
            data_file = os.path.join(data_dir, "integration.pkl")
            if os.path.exists(data_file):
                try:
                    print(f"📊 Computing stats from {data_file}")
                    data = torch.load(data_file, map_location='cpu')
                    
                    # Flatten data to compute statistics
                    qpos_flat = data['qpos_data'].reshape(-1, 14)
                    action_flat = data['action_data'].reshape(-1, 14)
                    
                    stats = {
                        'qpos_mean': qpos_flat.mean(dim=0).to(self.device),
                        'qpos_std': qpos_flat.std(dim=0).to(self.device) + 1e-6,
                        'action_mean': action_flat.mean(dim=0).to(self.device),
                        'action_std': action_flat.std(dim=0).to(self.device) + 1e-6,
                    }
                    
                    print(f"✅ Stats computed from data: qpos_mean={stats['qpos_mean'].mean():.4f}")
                    return stats
                    
                except Exception as e:
                    print(f"⚠️  Failed to load {data_file}: {e}")
                    continue
        
        # Fallback to default stats
        print(f"⚠️  Using default statistics")
        return {
            'qpos_mean': torch.zeros(14).to(self.device),
            'qpos_std': torch.ones(14).to(self.device),
            'action_mean': torch.zeros(14).to(self.device),
            'action_std': torch.ones(14).to(self.device)
        }
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint directory"""
        # 尝试多个可能的checkpoint路径
        possible_ckpt_paths = [
            "manibox/ManiBox/ckpt",  # 在ManiBox目录下运行
            "ckpt",  # 直接在当前目录
            "policy/ManiBox/manibox/ManiBox/ckpt",  # 在根目录下运行
            os.path.join(os.path.dirname(__file__), "manibox/ManiBox/ckpt"),  # 相对于这个文件的路径
        ]
        
        ckpt_base = None
        for path in possible_ckpt_paths:
            if os.path.exists(path):
                ckpt_base = path
                break
        
        if ckpt_base is None:
            # 打印调试信息
            current_dir = os.getcwd()
            script_dir = os.path.dirname(__file__)
            print(f"🔍 调试信息:")
            print(f"   当前目录: {current_dir}")
            print(f"   脚本目录: {script_dir}")
            print(f"   尝试的路径: {possible_ckpt_paths}")
            raise FileNotFoundError(f"Checkpoint directory not found in any of: {possible_ckpt_paths}")
        
        # Find directories starting with date
        ckpt_dirs = [d for d in os.listdir(ckpt_base) if d.startswith('2025')]
        if not ckpt_dirs:
            raise FileNotFoundError(f"No checkpoint directories found in {ckpt_base}")
        
        # Use the latest one
        latest_ckpt = sorted(ckpt_dirs)[-1]
        ckpt_dir = os.path.join(ckpt_base, latest_ckpt)
        
        print(f"📁 Using checkpoint directory: {ckpt_dir}")
        return ckpt_dir
    
    def reset_hidden_state(self):
        """Reset RNN hidden state for new episode"""
        # Use the model's reset_recur method
        batch_size = 1  # Single episode inference
        self.model.reset_recur(batch_size, self.device)
        print("🔄 RNN hidden state reset for new episode")
        
    def update_obs(self, obs):
        """Update observation cache"""
        self.obs_cache.append(obs)
    
    def encode_observation(self, observation):
        """Encode raw observation into model input format"""
        # Extract joint positions (qpos) from RoboTwin format
        if 'joint_action' in observation and 'vector' in observation['joint_action']:
            # RoboTwin format: observation['joint_action']['vector']
            qpos = observation['joint_action']['vector']
        elif 'qpos' in observation:
            # Direct format: observation['qpos']
            qpos = observation['qpos']
        else:
            raise KeyError("Could not find joint positions in observation. Expected 'joint_action.vector' or 'qpos'")
        
        # Extract camera images from RoboTwin format
        if 'observation' in observation:
            # RoboTwin format: observation['observation'][camera_name]['rgb']
            obs_data = observation['observation']
            cam_high = obs_data.get('head_camera', {}).get('rgb')
            cam_left_wrist = obs_data.get('left_camera', {}).get('rgb') 
            cam_right_wrist = obs_data.get('right_camera', {}).get('rgb')
        else:
            # Direct format: observation[camera_name]
            cam_high = observation.get('cam_high')
            cam_left_wrist = observation.get('cam_left_wrist')
            cam_right_wrist = observation.get('cam_right_wrist')
        
        # Process images with YOLO to get bounding box features
        bbox_features = self.yolo_processor.process(cam_high, cam_left_wrist, cam_right_wrist)
        
        # Convert to tensors
        if isinstance(qpos, np.ndarray):
            qpos_tensor = torch.from_numpy(qpos).float().to(self.device)
        else:
            qpos_tensor = torch.tensor(qpos).float().to(self.device)
        
        # Ensure qpos has correct shape (14,)
        if qpos_tensor.shape[0] != 14:
            raise ValueError(f"Expected qpos to have 14 dimensions, got {qpos_tensor.shape[0]}")
        
        bbox_tensor = bbox_features.float().to(self.device)
        
        # Normalize qpos
        qpos_normalized = (qpos_tensor - self.stats['qpos_mean']) / self.stats['qpos_std']
        
        return {
            'qpos': qpos_normalized,
            'bbox_features': bbox_tensor
        }
    
    def get_action(self):
        """Get action from the real RNN model"""
        if len(self.obs_cache) == 0:
            raise ValueError("Observation cache is empty")
        
        # Get latest observation
        latest_obs = self.obs_cache[-1]
        
        # Prepare model inputs
        robot_state = latest_obs['qpos']  # (14,)
        image_data = latest_obs['bbox_features']  # (batch_size, expected_dim)
        
        print(f"📊 Model input shapes - robot_state: {robot_state.shape}, image_data: {image_data.shape}")
        
        # Ensure correct batch dimensions for RNN inference
        if robot_state.dim() == 1:
            robot_state = robot_state.unsqueeze(0)  # (1, 14)
        if image_data.dim() == 1:
            image_data = image_data.unsqueeze(0)  # (1, bbox_dim)
        
        # Call the actual RNN model using the standard interface
        # Based on Clean_RNN.py inference example
        with torch.no_grad():
            action = self.model(
                image=image_data,           # (1, bbox_dim) 
                depth_image=None,           # Not using depth
                robot_state=robot_state,    # (1, 14)
                next_actions=None,          # Not used in inference
                next_actions_is_pad=None,   # Not used in inference
                actions=None,               # None means inference mode
                action_is_pad=None          # Not used in inference
            )
        
        # action shape should be (1, 14), extract the first batch
        if action.dim() > 1:
            action = action[0]  # (14,)
        
        # Convert to numpy and apply safety clipping
        action_np = action.cpu().numpy()
        action_np = np.clip(action_np, -2.0, 2.0)  # Safety clipping
        
        print(f"🎯 Generated action: shape={action_np.shape}, range=[{action_np.min():.3f}, {action_np.max():.3f}]")
        
        return [action_np]  # Return list of actions


def encode_obs(observation):
    """Post-Process Observation"""
    # Keep observation as is, processing happens in the model
    return observation


def get_model(usr_args):
    """Load and return ManiBox RNN model"""
    print(f"🚀 Initializing ManiBox RNN model...")
    model = ManiBoxRNNModel(usr_args)
    return model


def eval(TASK_ENV, model, observation):
    """
    Evaluation function for ManiBox RNN policy
    """
    # Encode observation
    obs = encode_obs(observation)
    
    # Get task instruction (if needed)
    instruction = TASK_ENV.get_instruction()
    
    # Process observation
    processed_obs = model.encode_observation(obs)
    
    # Force update observation at first frame
    if len(model.obs_cache) == 0:
        model.update_obs(processed_obs)
    
    # Get actions from model
    actions = model.get_action()
    
    # Execute each action
    for action in actions:
        # Execute joint control
        TASK_ENV.take_action(action, action_type='qpos')
        
        # Get new observation
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        processed_obs = model.encode_observation(obs)
        
        # Update observation cache
        model.update_obs(processed_obs)


def reset_model(model):
    """Reset model state for new episode"""
    # Clear observation cache
    model.obs_cache.clear()
    
    # Reset RNN hidden state using the actual model's reset method
    model.reset_hidden_state()
    
    # Reset YOLO processor for new episode
    model.yolo_processor.reset_new_episode()
    
    print("🔄 Model completely reset for new episode")


if __name__ == "__main__":
    # Test model loading
    test_args = {
        'task_name': 'grasp_apple',
        'ckpt_setting': 'policy_best',
        'objects': ['apple']
    }
    
    try:
        model = get_model(test_args)
        print("✅ Model loading test successful!")
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
