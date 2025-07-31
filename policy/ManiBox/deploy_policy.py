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
from PIL import Image, ImageDraw

# Add multiple possible paths for robust importing
current_dir = os.path.dirname(__file__)
manibox_dir = os.path.join(current_dir, "manibox")
manibox_manibox_dir = os.path.join(current_dir, "manibox/ManiBox")
manibox_policy_dir = os.path.join(current_dir, "manibox/ManiBox/policy")

for path in [manibox_dir, manibox_manibox_dir, manibox_policy_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Also ensure we can import from the current working directory
cwd = os.getcwd()
if "RoboTwin" in cwd and "policy/ManiBox" in cwd:
    # We're in ManiBox directory, add the manibox paths
    robotwin_root = cwd.replace("/policy/ManiBox", "")
    manibox_policy_dir = os.path.join(robotwin_root, "policy/ManiBox/manibox")
    if manibox_policy_dir not in sys.path:
        sys.path.insert(0, manibox_policy_dir)

try:
    from yolo_process_data import YoloProcessDataByTimeStep
except ImportError:
    try:
        from ManiBox.yolo_process_data import YoloProcessDataByTimeStep
    except ImportError:
        # Last resort: try absolute import from manibox directory
        manibox_abs_dir = os.path.abspath(os.path.join(current_dir, "manibox"))
        if manibox_abs_dir not in sys.path:
            sys.path.insert(0, manibox_abs_dir)
        from ManiBox.yolo_process_data import YoloProcessDataByTimeStep


def make_policy(policy_class, policy_config, pretrain_ckpt_dir):
    if len(pretrain_ckpt_dir) != 0:
        pretrain_ckpt_dir = os.path.join(pretrain_ckpt_dir, "policy_best.ckpt")
    if policy_class == 'ACT':
        from policy.ACTPolicy import ACTPolicy
        policy = ACTPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            state_dict = torch.load(pretrain_ckpt_dir)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key in ["model.is_pad_head.weight", "model.is_pad_head.bias"]:
                    continue
                if policy_config['num_next_action'] == 0 and key in ["model.input_proj_next_action.weight",
                                                                     "model.input_proj_next_action.bias"]:
                    continue
                new_state_dict[key] = value
            loading_status = policy.load_state_dict(new_state_dict)
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'CNNMLP':
        from policy.CNNMLPPolicy import CNNMLPPolicy
        policy = CNNMLPPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'HistoryCNNMLP':
        from policy.HistoryCNNMLPPolicy import HistoryCNNMLPPolicy
        policy = HistoryCNNMLPPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'CNNRNN':
        from policy.CNNRNNPolicy import CNNRNNPolicy
        policy = CNNRNNPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'FPNRNN':
        from policy.FPNRNNPolicy import FPNRNNPolicy
        policy = FPNRNNPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'RNN':
        try:
            from policy.Clean_RNN import RNNPolicy
        except ImportError:
            try:
                from ManiBox.policy.Clean_RNN import RNNPolicy
            except ImportError:
                from Clean_RNN import RNNPolicy
        policy = RNNPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'DiffusionState':
        from policy.DiffusionStatePolicy import DiffusionStatePolicy
        policy = DiffusionStatePolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'Diffusion':
        from policy.DiffusionPolicy import DiffusionPolicy
        policy = DiffusionPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    else:
        raise NotImplementedError
    return policy



class ManiBoxRNNModel:
    """ManiBox RNN model for deployment in evaluation environment"""
    
    def __init__(self, usr_args):
        """Initialize ManiBox RNN model"""
        # Extract configuration
        self.task_name = usr_args.get('task_name', 'grasp_apple')
        self.ckpt_setting = usr_args.get('ckpt_setting', 'policy_best')
        self.device = usr_args.get('device', 'cuda:0')
        
        # Construct checkpoint path
        self.ckpt_dir = self._find_latest_checkpoint()
        
        print(f"🤖 Loading ManiBox RNN model from: {self.ckpt_dir}")
        
        # Initialize YOLO processor first
        objects_raw = usr_args.get('objects', ['apple'])
        max_detections = usr_args.get('max_detections_per_object', 2)  # Default to 2 for multiple same objects
        
        # Parse objects if it's a string (from script/eval_policy.py)
        if isinstance(objects_raw, str):
            import ast
            try:
                objects_names = ast.literal_eval(objects_raw)
            except Exception as e:
                print(f"⚠️ Failed to parse objects string '{objects_raw}', using fallback: {e}")
                objects_names = ['bottle']  # fallback
        else:
            objects_names = objects_raw
            
        print(f"🔧 Debug: objects_raw = {objects_raw}, type = {type(objects_raw)}")
        print(f"🔧 Debug: objects_names = {objects_names}, max_detections = {max_detections}")
        
        self.yolo_processor = YoloProcessDataByTimeStep(
            objects_names=objects_names,
            max_detections_per_object=max_detections  # Enable multiple detections of same object
        )
        
        # CRITICAL: Ensure class variables are set before policy creation
        # This is needed because train.py imports may create modules that read these values
        YoloProcessDataByTimeStep.objects_names = objects_names
        YoloProcessDataByTimeStep.total_detections_class = self.yolo_processor.total_detections
        print(f"🔧 Force update class variables: objects={YoloProcessDataByTimeStep.objects_names}, total_detections={YoloProcessDataByTimeStep.total_detections_class}")
        
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
        
        # Initialize visualization settings
        self.enable_visualization = usr_args.get('enable_visualization', True)
        self.viz_save_path = usr_args.get('viz_save_path', './eval_visualization')
        self.step_count = 0
        
        # Initialize video frames cache (no individual image saving)
        self.video_frames_cache = {
            'cam_high': [],
            'cam_left_wrist': [],
            'cam_right_wrist': []
        }
        
        if self.enable_visualization:
            os.makedirs(self.viz_save_path, exist_ok=True)
            print(f"📹 Video visualization enabled, saving to: {self.viz_save_path}")
        
        print(f"🎯 ManiBox RNN model initialization complete")
        print(f"   Device: {self.device}")
        print(f"   Objects: {objects_names}")
        print(f"   Max detections per object: {max_detections}")
        print(f"   Total detections: {self.yolo_processor.total_detections}")
        print(f"   Task: {self.task_name}")
        print(f"   Expected bbox dim: {self.yolo_processor.total_detections * 3 * 4}")
    
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
            'rnn_layers': 5,
            'rnn_hidden_dim': 1024,
            'actor_hidden_dim': 1024,
            'policy_class': 'RNN',
            'gradient_accumulation_steps': 1
        }
        
        return policy_config
    
    def _load_stats(self):
        """Load dataset statistics for normalization"""
        # Directly use default statistics
        print(f"⚠️  Using default statistics")
        return {
            'qpos_mean': torch.zeros(14).to(self.device),
            'qpos_std': torch.ones(14).to(self.device),
            'action_mean': torch.zeros(14).to(self.device),
            'action_std': torch.ones(14).to(self.device)
        }
    
    def _compute_stats_from_data(self):
        """Compute statistics from processed data files"""
        # First, try to find data matching the current task
        # Convert task name from underscore to hyphen format (e.g., pick_diverse_bottles -> pick-diverse-bottles)
        task_name_hyphen = self.task_name.replace('_', '-')
        task_specific_dirs = [
            f"processed_data/manibox-{task_name_hyphen}",  # 在ManiBox目录下
            f"policy/ManiBox/processed_data/manibox-{task_name_hyphen}",  # 在根目录下
            os.path.join(os.path.dirname(__file__), f"processed_data/manibox-{task_name_hyphen}"),  # 相对于脚本文件的路径
        ]
        
        # Try task-specific data first
        for data_dir in task_specific_dirs:
            data_file = os.path.join(data_dir, "integration.pkl")
            if os.path.exists(data_file):
                try:
                    print(f"📊 Computing stats from task-specific data: {data_file}")
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
                    
                    print(f"✅ Stats computed from task-specific data: qpos_mean={stats['qpos_mean'].mean():.4f}")
                    return stats
                    
                except Exception as e:
                    print(f"⚠️  Failed to load task-specific data {data_file}: {e}")
                    continue
        
        # If task-specific data not found, try fallback data
        fallback_data_dirs = [
            "processed_data/manibox-beat_block_hammer",  # 在ManiBox目录下
            "policy/ManiBox/processed_data/manibox-beat_block_hammer",  # 在根目录下
            "processed_data/manibox-grasp_apple",  # 在ManiBox目录下
            "policy/ManiBox/processed_data/manibox-grasp_apple",  # 在根目录下
            os.path.join(os.path.dirname(__file__), "processed_data/manibox-beat_block_hammer"),
            os.path.join(os.path.dirname(__file__), "processed_data/manibox-grasp_apple"),
        ]
        
        for data_dir in fallback_data_dirs:
            data_file = os.path.join(data_dir, "integration.pkl")
            if os.path.exists(data_file):
                try:
                    print(f"📊 Computing stats from fallback data: {data_file}")
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
                    
                    print(f"✅ Stats computed from fallback data: qpos_mean={stats['qpos_mean'].mean():.4f}")
                    return stats
                    
                except Exception as e:
                    print(f"⚠️  Failed to load fallback data {data_file}: {e}")
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
        # 检查是否在配置中指定了具体的checkpoint路径
        if hasattr(self, 'ckpt_setting') and self.ckpt_setting and self.ckpt_setting != 'null':
            # 如果指定了具体的checkpoint，直接使用
            if os.path.isabs(self.ckpt_setting):
                # 绝对路径
                ckpt_dir = self.ckpt_setting
            else:
                # 相对路径，尝试多个可能的base路径
                possible_base_paths = [
                    "ckpt",  # 在ManiBox目录下运行
                    "policy/ManiBox/ckpt",  # 在根目录下运行
                    os.path.join(os.path.dirname(__file__), "ckpt"),  # 相对于脚本的路径
                    "manibox/ckpt",  # manibox子目录下的ckpt
                    "policy/ManiBox/manibox/ckpt",  # 从根目录访问manibox/ckpt
                    "manibox/ManiBox/ckpt",  # 在manibox/ManiBox目录下的ckpt
                ]
                
                ckpt_dir = None
                print(f"🔍 Debug: Looking for checkpoint '{self.ckpt_setting}' in base paths:")
                for base_path in possible_base_paths:
                    full_path = os.path.join(base_path, self.ckpt_setting)
                    exists = os.path.exists(full_path)
                    print(f"   {base_path} -> {full_path} (exists: {exists})")
                    if exists:
                        ckpt_dir = full_path
                        print(f"   ✅ Found: {ckpt_dir}")
                        break
                
                if ckpt_dir is None:
                    raise FileNotFoundError(f"Specified checkpoint '{self.ckpt_setting}' not found in any base path: {possible_base_paths}")
            
            print(f"📁 Using specified checkpoint directory: {ckpt_dir}")
            return ckpt_dir
        
        # 如果没有指定，使用原来的自动查找逻辑
        print("🔍 No specific checkpoint specified, searching for latest checkpoint...")
        
        # 尝试多个可能的checkpoint路径
        possible_ckpt_paths = [
            "ckpt",  # 在ManiBox目录下运行
            "policy/ManiBox/ckpt",  # 在根目录下运行
            os.path.join(os.path.dirname(__file__), "ckpt"),  # 相对于这个文件的路径
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
    
    def visualize_detection(self, cameras, bboxes_list):
        """Visualize detected bboxes on camera images and cache frames for video
        
        Args:
            cameras: dict with camera images {'head_camera': img, 'left_camera': img, 'right_camera': img}
            bboxes_list: list of bboxes for each camera in format [(cam_num, objects_num, 4)]
        """
        if not self.enable_visualization:
            return
            
        camera_names = ['head_camera', 'left_camera', 'right_camera']
        camera_mapping = {
            'head_camera': 'cam_high',
            'left_camera': 'cam_left_wrist', 
            'right_camera': 'cam_right_wrist'
        }
        
        for i, cam_name in enumerate(camera_names):
            if cam_name in cameras and cameras[cam_name] is not None:
                image = cameras[cam_name]
                
                # Convert image to proper format if needed
                if isinstance(image, torch.Tensor):
                    if image.shape[0] == 3:  # CHW format
                        image = image.permute(1, 2, 0)  # Convert to HWC
                    if image.max() <= 1.0:  # Normalize to 0-255
                        image = (image * 255).clamp(0, 255).byte()
                    image = image.cpu().numpy()
                elif isinstance(image, np.ndarray):
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                
                # Extract bboxes for this camera
                if i < len(bboxes_list):
                    bboxes = bboxes_list[i]  # Shape: (objects_num, 4)
                    
                    # Convert to PIL Image for drawing
                    pil_image = Image.fromarray(image, 'RGB')
                    draw = ImageDraw.Draw(pil_image)
                    
                    # Draw bboxes
                    H, W = image.shape[:2]
                    colors = ['red', 'blue', 'green', 'yellow', 'purple']
                    
                    for obj_idx, bbox in enumerate(bboxes):
                        if not np.allclose(bbox, [0, 0, 0, 0]):  # Skip empty bboxes
                            # Convert normalized coordinates to pixel coordinates
                            x1, y1, x2, y2 = bbox
                            x1, x2 = x1 * W, x2 * W
                            y1, y2 = y1 * H, y2 * H
                            
                            # Draw rectangle
                            color = colors[obj_idx % len(colors)]
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                            
                            # Draw label
                            if obj_idx < len(self.yolo_processor.objects_names):
                                label = self.yolo_processor.objects_names[obj_idx]
                                draw.text((x1, y1-20), label, fill=color)
                    
                    # Cache frame for video generation (convert PIL to numpy array)
                    frame_array = np.array(pil_image)
                    camera_key = camera_mapping[cam_name]
                    self.video_frames_cache[camera_key].append(frame_array)
        
        self.step_count += 1
    
    def create_visualization_video(self, episode_id=0, fps=10):
        """Create video directly from cached frames (no intermediate images)"""
        if not self.enable_visualization:
            return
            
        try:
            import imageio
            
            camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
            
            for cam_name in camera_names:
                # Get cached frames for this camera
                frames = self.video_frames_cache.get(cam_name, [])
                
                if frames and len(frames) > 0:
                    # Create video directly from cached frames
                    video_path = os.path.join(self.viz_save_path, f"episode_{episode_id:03d}_{cam_name}_bbox_video.mp4")
                    
                    print(f"🎬 Creating video: {video_path} with {len(frames)} frames...")
                    
                    # Ensure frames are valid
                    valid_frames = []
                    for i, frame in enumerate(frames):
                        if frame is not None and frame.size > 0:
                            valid_frames.append(frame)
                        else:
                            print(f"⚠️ Skipping invalid frame {i}")
                    
                    if len(valid_frames) < 2:
                        print(f"⚠️ Too few valid frames ({len(valid_frames)}) for {cam_name}, skipping video creation")
                        continue
                    
                    # Create video with better codec settings
                    with imageio.get_writer(
                        video_path, 
                        fps=fps,
                        codec='libx264',
                        quality=8,
                        pixelformat='yuv420p'
                    ) as writer:
                        for frame in valid_frames:
                            writer.append_data(frame)
                    
                    # Verify video file was created successfully
                    if os.path.exists(video_path):
                        file_size = os.path.getsize(video_path)
                        if file_size > 1000:  # At least 1KB
                            print(f"✅ Created bbox video: {video_path} ({len(valid_frames)} frames, {file_size} bytes)")
                        else:
                            print(f"❌ Video file too small: {video_path} ({file_size} bytes), possible corruption")
                    else:
                        print(f"❌ Video file not created: {video_path}")
                        
                else:
                    print(f"⚠️ No frames cached for camera: {cam_name} (cached: {len(frames)})")
            
        except ImportError:
            print("⚠️ imageio not available, skipping video creation")
        except Exception as e:
            print(f"⚠️ Error creating visualization video: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_video_cache(self):
        """Clear video frames cache for new episode"""
        for cam_name in self.video_frames_cache:
            self.video_frames_cache[cam_name].clear()
        print("🗑️ Video frames cache cleared")
    
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
        
        # Debug: Print bbox_features shape
        print(f"🔍 Debug: bbox_features shape = {bbox_features.shape}")
        print(f"🔍 Debug: bbox_features content = {bbox_features}")
        
        # Use only first 12 dimensions of bbox features for inference
        # This matches the training format where we use 12-dim bbox (2 cameras * 1 object * 4 coordinates)
        if bbox_features.shape[0] == 24:
            bbox_features = bbox_features[:12]
            print(f"🔄 Using first 12 dimensions of bbox features for inference (shape: {bbox_features.shape})")
        elif bbox_features.shape[0] == 12:
            print(f"✅ Bbox features already have 12 dimensions (shape: {bbox_features.shape})")
        else:
            print(f"⚠️ Unexpected bbox dimensions: {bbox_features.shape[0]}, using first 12")
            bbox_features = bbox_features[:12]
        
        # Now we have 12-dim bbox_features + 14-dim qpos = 26-dim total input
        # This matches the training format (12 + 14 = 26)
        
        # Extract bboxes for visualization if enabled
        if self.enable_visualization and cam_high is not None:
            # Get raw bbox coordinates for visualization
            cameras_for_viz = [cam_high, cam_left_wrist, cam_right_wrist]
            bboxes_for_viz = self.yolo_processor.parallel_detect_bounding_boxes(cameras_for_viz)
            
            # Organize cameras for visualization
            camera_dict = {}
            if cam_high is not None:
                camera_dict['head_camera'] = cam_high
            if cam_left_wrist is not None:
                camera_dict['left_camera'] = cam_left_wrist  
            if cam_right_wrist is not None:
                camera_dict['right_camera'] = cam_right_wrist
                
            # Call visualization
            self.visualize_detection(camera_dict, bboxes_for_viz)
        
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
        """Get action from the real RNN model - Fixed to match Isaac Lab pattern"""
        if len(self.obs_cache) == 0:
            raise ValueError("Observation cache is empty")
        
        # Get latest observation
        latest_obs = self.obs_cache[-1]
        
        # Prepare model inputs
        robot_state = latest_obs['qpos']  # (14,)
        image_data = latest_obs['bbox_features']  # (bbox_dim,)
        
        print(f"📊 Model input shapes - robot_state: {robot_state.shape}, image_data: {image_data.shape}")
        
        # Ensure correct batch dimensions for RNN inference
        if robot_state.dim() == 1:
            robot_state = robot_state.unsqueeze(0)  # (1, 14)
        if image_data.dim() == 1:
            image_data = image_data.unsqueeze(0)  # (1, bbox_dim)
        
        # Call the actual RNN model using the standard interface
        # Based on Isaac Lab inference pattern
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
    Fixed to match Isaac Lab inference pattern
    """
    # Encode observation
    obs = encode_obs(observation)
    
    # Process observation
    processed_obs = model.encode_observation(obs)
    
    # Update observation cache (this maintains RNN state continuity)
    model.update_obs(processed_obs)
    
    # Get single action from model (not multiple actions)
    action = model.get_action()
    
    # Execute single action (not loop through multiple actions)
    if action is not None and len(action) > 0:
        TASK_ENV.take_action(action[0], action_type='qpos')
    
    # Check if episode should end and finalize video
    # This happens when eval_success is True or step limit is reached
    if hasattr(model, 'enable_visualization') and model.enable_visualization:
        episode_ending = (TASK_ENV.eval_success or 
                         TASK_ENV.take_action_cnt >= TASK_ENV.step_lim - 1)
        
        if episode_ending and hasattr(model, '_episode_count'):
            print(f"🎬 Episode {model._episode_count} ending, finalizing video...")
            finalize_episode(model, model._episode_count)
            # Mark this episode as finalized to avoid duplicate calls
            model._episode_finalized = True


def reset_model(model):
    """Reset model state for new episode"""
    # Finalize previous episode's video before resetting (if not already finalized)
    if hasattr(model, 'enable_visualization') and model.enable_visualization:
        if hasattr(model, '_episode_count'):
            # Only finalize if not already done
            if not getattr(model, '_episode_finalized', False):
                print(f"🎬 Finalizing episode {model._episode_count} video on reset...")
                finalize_episode(model, model._episode_count)
            model._episode_count += 1
            model._episode_finalized = False  # Reset for new episode
        else:
            # First episode
            model._episode_count = 1
            model._episode_finalized = False
    
    # Clear observation cache
    model.obs_cache.clear()
    
    # Reset RNN hidden state using the actual model's reset method
    model.reset_hidden_state()
    
    # Reset YOLO processor for new episode
    model.yolo_processor.reset_new_episode()
    
    # Reset visualization for new episode
    if hasattr(model, 'enable_visualization') and model.enable_visualization:
        model.step_count = 0
        model.clear_video_cache()
        print(f"🔄 Visualization reset for episode {model._episode_count}")
    
    print("🔄 Model completely reset for new episode")


def finalize_episode(model, episode_id=0):
    """Finalize episode by creating visualization video"""
    if hasattr(model, 'enable_visualization') and model.enable_visualization:
        print(f"🎬 Finalizing episode {episode_id}, creating visualization video...")
        
        # Check if we have frames to create video
        total_frames = sum(len(frames) for frames in model.video_frames_cache.values())
        
        if total_frames > 0:
            model.create_visualization_video(episode_id=episode_id)
            print(f"✅ Episode {episode_id} visualization completed")
        else:
            print(f"⚠️ No frames cached for episode {episode_id}, skipping video creation")


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
