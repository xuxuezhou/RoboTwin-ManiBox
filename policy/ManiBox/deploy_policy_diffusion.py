#!/usr/bin/env python3
"""
ManiBox Policy Deployment
Deploy ManiBox SimpleBBoxDiffusion policy for evaluation with video recording
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
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        sys.path.insert(0, os.path.join(current_dir, 'policy'))
        from manibox.ManiBox.policy.DiffusionPolicy import DiffusionPolicy
        policy = DiffusionPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'BBoxDiffusion':
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        sys.path.insert(0, os.path.join(current_dir, 'policy'))
        from policy.BBoxDiffusionPolicy import BBoxDiffusionPolicy
        policy = BBoxDiffusionPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'SimpleBBoxDiffusion':
        import sys
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        sys.path.insert(0, os.path.join(current_dir, 'policy'))
        from manibox.ManiBox.policy.SimpleBBoxDiffusionPolicy import SimpleBBoxDiffusionPolicy
        policy = SimpleBBoxDiffusionPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.load_state_dict(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    else:
        raise NotImplementedError
    return policy



class ManiBoxDiffusionModel:
    """ManiBox SimpleBBoxDiffusion model for deployment in evaluation environment"""
    
    def __init__(self, usr_args):
        """Initialize ManiBox SimpleBBoxDiffusion model"""
        # Extract configuration
        self.task_name = usr_args.get('task_name', 'grasp_apple')
        self.ckpt_setting = usr_args.get('ckpt_setting', 'policy_best')
        self.device = usr_args.get('device', 'cuda:0')
        
        # Construct checkpoint path
        self.ckpt_dir = self._find_latest_checkpoint()
        
        print(f"🤖 Loading ManiBox SimpleBBoxDiffusion model from: {self.ckpt_dir}")
        
        # Initialize YOLO processor first
        objects_raw = usr_args.get('objects', ['bottle'])
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
        
        # Load model configuration from checkpoint
        config_path = os.path.join(self.ckpt_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                checkpoint_config = json.load(f)
            policy_class = checkpoint_config.get('policy_class', 'Diffusion')
            print(f"📋 Loaded policy class from checkpoint: {policy_class}")
        else:
            policy_class = 'Diffusion'
            print(f"⚠️  No config.json found, using default policy class: {policy_class}")
        
        # Create policy configuration
        self.policy_config = self._create_policy_config(usr_args, objects_names, policy_class)
        
        # Create and load the actual model
        self.model = make_policy(policy_class, self.policy_config, self.ckpt_dir)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Real ManiBox SimpleBBoxDiffusion model loaded successfully!")
        
        # Load statistics for normalization
        self.stats = self._load_stats()
        
        # Initialize observation cache for diffusion policy
        self.obs_cache = deque(maxlen=1)  # Only need current observation for diffusion
        
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
        
        # Initialize action logging
        self.action_log_file = None
        self.action_log_path = None
        self.timestep = 0
        
        print(f"🎯 ManiBox SimpleBBoxDiffusion model initialization complete")
        print(f"   Device: {self.device}")
        print(f"   Objects: {objects_names}")
        print(f"   Max detections per object: {max_detections}")
        print(f"   Total detections: {self.yolo_processor.total_detections}")
        print(f"   Task: {self.task_name}")
        print(f"   Expected bbox dim: {self.yolo_processor.total_detections * 3 * 4}")
    
    def _create_policy_config(self, usr_args, objects_names, policy_class='Diffusion'):
        """Create policy configuration for diffusion model"""
        # Default camera setup
        camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
        
        policy_config = {
            'lr': 1e-4,
            'lr_backbone': 1e-4,
            'epochs': 100,
            'train_loader_len': 100,  # placeholder
            'warmup_ratio': 0.1,
            'use_scheduler': 'cos',
            'backbone': 'resnet18',
            'masks': False,
            'weight_decay': 1e-4,
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
            'policy_class': policy_class,
            'gradient_accumulation_steps': 1,
            # Diffusion specific parameters
            'action_horizon': 8,
            'observation_horizon': 1,
            'num_inference_timesteps': 20,
            'num_objects': len(objects_names)
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
        # Convert task name to dataset name format (replace underscores with hyphens)
        dataset_name = self.task_name.replace('_', '-')
        
        # Look for processed data files in different possible locations
        possible_data_dirs = [
            # f"processed_data/manibox-{dataset_name}",  # 在ManiBox目录下
            f"policy/ManiBox/processed_data/manibox-{dataset_name}",  # 在根目录下
            f"processed_data/manibox-{self.task_name}",  # 在ManiBox目录下（原始格式）
            f"policy/ManiBox/processed_data/manibox-{self.task_name}",  # 在根目录下（原始格式）
            "processed_data/manibox-beat_block_hammer",  # 在ManiBox目录下
            "policy/ManiBox/processed_data/manibox-beat_block_hammer",  # 在根目录下
            "processed_data/manibox-grasp_apple",  # 在ManiBox目录下
            "policy/ManiBox/processed_data/manibox-grasp_apple",  # 在根目录下
            # 相对于脚本文件的路径
            os.path.join(os.path.dirname(__file__), f"processed_data/manibox-{dataset_name}"),
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
        # 检查是否在配置中指定了具体的checkpoint路径
        if hasattr(self, 'ckpt_setting') and self.ckpt_setting and self.ckpt_setting != 'null':
            # 如果指定了具体的checkpoint，直接使用
            if os.path.isabs(self.ckpt_setting):
                # 绝对路径
                ckpt_dir = self.ckpt_setting
            else:
                # 相对路径，尝试多个可能的base路径
                possible_base_paths = [
                    "policy/ManiBox/ckpt",  # 在根目录下运行
                    os.path.join(os.path.dirname(__file__), "ckpt"),  # 相对于脚本的路径
                    "ckpt",  # 在ManiBox目录下运行
                    "manibox/ckpt",  # manibox子目录下的ckpt
                    "policy/ManiBox/manibox/ckpt",  # 从根目录访问manibox/ckpt
                    "eval_result/pick_diverse_bottles/ManiBox/demo_clean/policy/ManiBox/ckpt",  # eval_result中的checkpoint
                    "../../ckpt",  # 从policy/ManiBox目录访问根目录的ckpt
                ]
                
                ckpt_dir = None
                for base_path in possible_base_paths:
                    full_path = os.path.join(base_path, self.ckpt_setting)
                    if os.path.exists(full_path):
                        # Check if policy_best.ckpt exists
                        ckpt_file = os.path.join(full_path, "policy_best.ckpt")
                        if os.path.exists(ckpt_file):
                            ckpt_dir = full_path
                            break
                        else:
                            print(f"⚠️  Found directory {full_path} but no policy_best.ckpt file")
                
                if ckpt_dir is None:
                    print(f"⚠️  Specified checkpoint '{self.ckpt_setting}' not found, trying to find an available Diffusion model...")
                    # Try to find any available Diffusion model
                    for base_path in possible_base_paths:
                        if os.path.exists(base_path):
                            ckpt_dirs = [d for d in os.listdir(base_path) if 'Diffusion' in d and os.path.exists(os.path.join(base_path, d, 'policy_best.ckpt'))]
                            if ckpt_dirs:
                                # Use the latest one
                                latest_ckpt = sorted(ckpt_dirs)[-1]
                                ckpt_dir = os.path.join(base_path, latest_ckpt)
                                print(f"✅ Found available Diffusion model: {ckpt_dir}")
                                break
                    
                    if ckpt_dir is None:
                        raise FileNotFoundError(f"No available Diffusion models found in any base path: {possible_base_paths}")
            
            print(f"📁 Using specified checkpoint directory: {ckpt_dir}")
            return ckpt_dir
        
        # 如果没有指定，使用原来的自动查找逻辑
        print("🔍 No specific checkpoint specified, searching for latest checkpoint...")
        
        # 尝试多个可能的checkpoint路径
        possible_ckpt_paths = [
            "policy/ManiBox/ckpt",  # 在根目录下运行，优先选择
            os.path.join(os.path.dirname(__file__), "ckpt"),  # 相对于这个文件的路径
            "ckpt",  # 在ManiBox目录下运行，最后选择
        ]
        
        print(f"🔍 Debug: Current working directory: {os.getcwd()}")
        print(f"🔍 Debug: Script directory: {os.path.dirname(__file__)}")
        print(f"🔍 Debug: Checking possible paths:")
        for i, path in enumerate(possible_ckpt_paths):
            exists = os.path.exists(path)
            print(f"   {i+1}. {path} - exists: {exists}")
            if exists:
                print(f"      Contents: {os.listdir(path)}")
        
        ckpt_base = None
        for path in possible_ckpt_paths:
            if os.path.exists(path):
                ckpt_base = path
                print(f"🔍 Debug: Found checkpoint base path: {ckpt_base}")
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
        
        print(f"🔍 Debug: Found checkpoint directories: {ckpt_dirs}")
        
        # 优先选择包含 SimpleBBoxDiffusion 的检查点
        simple_bbox_dirs = [d for d in ckpt_dirs if 'SimpleBBoxDiffusion' in d]
        print(f"🔍 Debug: SimpleBBoxDiffusion directories: {simple_bbox_dirs}")
        
        if simple_bbox_dirs:
            # 如果有 SimpleBBoxDiffusion 检查点，使用最新的
            latest_ckpt = sorted(simple_bbox_dirs)[-1]
            print(f"🔍 Debug: Selected SimpleBBoxDiffusion checkpoint: {latest_ckpt}")
        else:
            # 否则使用最新的检查点
            latest_ckpt = sorted(ckpt_dirs)[-1]
            print(f"🔍 Debug: No SimpleBBoxDiffusion found, using latest: {latest_ckpt}")
            
        ckpt_dir = os.path.join(ckpt_base, latest_ckpt)
        
        print(f"📁 Using checkpoint directory: {ckpt_dir}")
        return ckpt_dir
    
    def reset_hidden_state(self):
        """Reset model state for new episode (not needed for diffusion but kept for compatibility)"""
        # Clear observation cache
        self.obs_cache.clear()
        print("🔄 Model state reset for new episode")
    
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
    
    def start_action_logging(self, log_path=None):
        """Start logging actions to file"""
        if log_path is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"action_log_{timestamp}.jsonl"
        
        self.action_log_path = log_path
        self.action_log_file = open(log_path, 'w')
        self.timestep = 0
        print(f"📝 Action logging started: {log_path}")
    
    def stop_action_logging(self):
        """Stop logging actions and close file"""
        if self.action_log_file is not None:
            self.action_log_file.close()
            self.action_log_file = None
            print(f"📝 Action logging stopped: {self.action_log_path}")
            self.action_log_path = None
    
    def encode_observation(self, observation):
        """Encode raw observation into model input format for SimpleBBoxDiffusion"""
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
        
        # Print raw bbox detection results
        print(f"📦 Raw BBox Features: shape={bbox_features.shape}")
        if hasattr(bbox_features, 'shape') and len(bbox_features.shape) > 0:
            bbox_flat = bbox_features.flatten()
            print(f"   Values: {bbox_flat[:12].tolist()}...")  # 显示前12个值
            
            # 检查并修正 bbox 坐标边界
            bbox_flat_clipped = torch.clamp(bbox_flat, 0.0, 1.0)
            if not torch.allclose(bbox_flat, bbox_flat_clipped):
                print(f"⚠️  BBox coordinates clipped: {torch.sum(bbox_flat != bbox_flat_clipped)} values out of bounds")
                bbox_features = bbox_flat_clipped.reshape(bbox_features.shape)
            
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
            
            # 解析 bbox 数据 (2个相机 × 1个检测 × 4个坐标)
            if bbox_flat.shape[0] >= 12:
                for cam_idx, cam_name in enumerate(['head', 'left_wrist']):
                    start_idx = cam_idx * 4  # Only first object per camera
                    bbox = bbox_flat[start_idx:start_idx+4].tolist()
                    bbox_clipped = bbox_flat_clipped[start_idx:start_idx+4].tolist()
                    if bbox != bbox_clipped:
                        print(f"   {cam_name}_cam, detection_0: {bbox} -> {bbox_clipped} (clipped)")
                    else:
                        print(f"   {cam_name}_cam, detection_0: {bbox}")
        
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
        """Get action from the SimpleBBoxDiffusion model"""
        if len(self.obs_cache) == 0:
            raise ValueError("Observation cache is empty")
        
        # Get latest observation
        latest_obs = self.obs_cache[-1]
        
        # Prepare model inputs for SimpleBBoxDiffusion
        robot_state = latest_obs['qpos']  # (14,)
        bbox_data = latest_obs['bbox_features']  # (bbox_dim,)
        
        print(f"📊 Model input shapes - robot_state: {robot_state.shape}, bbox_data: {bbox_data.shape}")
        
        # Ensure correct batch dimensions for diffusion inference
        if robot_state.dim() == 1:
            robot_state = robot_state.unsqueeze(0)  # (1, 14)
        if bbox_data.dim() == 1:
            bbox_data = bbox_data.unsqueeze(0)  # (1, bbox_dim)
        
        # Add time dimension for diffusion model
        if bbox_data.dim() == 2:
            bbox_data = bbox_data.unsqueeze(1)  # (1, 1, bbox_dim)
        if robot_state.dim() == 2:
            robot_state = robot_state.unsqueeze(1)  # (1, 1, 14)
        
        # Call the SimpleBBoxDiffusion model using predict_action method
        with torch.no_grad():
            action = self.model.predict_action(bbox_data, robot_state)
        
        # action shape should be (1, action_horizon, action_dim), extract the first action
        if action.dim() == 3:
            action = action[0, 0]  # (action_dim,) - take first action from first batch
        elif action.dim() == 2:
            action = action[0]  # (action_dim,) - take first action
        
        # Ensure action has correct shape (14,)
        if action.shape[0] != 14:
            print(f"⚠️ Warning: Expected action to have 14 dimensions, got {action.shape[0]}")
            # Pad or truncate to 14 dimensions if needed
            if action.shape[0] < 14:
                action = torch.cat([action, torch.zeros(14 - action.shape[0], device=action.device)])
            else:
                action = action[:14]
        
        # Denormalize action using dataset statistics
        if hasattr(self, 'stats') and 'action_mean' in self.stats and 'action_std' in self.stats:
            action_denorm = action * self.stats['action_std'] + self.stats['action_mean']
            print(f"🔄 Action denormalized using dataset stats")
        else:
            action_denorm = action
            print(f"⚠️ No dataset stats available, using raw action")
        
        # Convert to numpy and apply safety clipping
        action_np = action_denorm.cpu().numpy()
        action_np = np.clip(action_np, -2.0, 2.0)  # Safety clipping
        
        print(f"🎯 Generated action: shape={action_np.shape}, range=[{action_np.min():.3f}, {action_np.max():.3f}]")
        
        return [action_np]  # Return list of actions


def encode_obs(observation):
    """Post-Process Observation"""
    # Print bounding box information if available
    if 'observation' in observation and 'bbox' in observation['observation']:
        bbox_data = observation['observation']['bbox']
        print(f"📦 Raw Bounding Box Data: shape={bbox_data.shape if hasattr(bbox_data, 'shape') else 'No shape'}")
        if hasattr(bbox_data, 'shape') and len(bbox_data.shape) > 0:
            print(f"   Values: {bbox_data.flatten()[:12]}...")  # 显示前12个值
    elif 'bbox' in observation:
        bbox_data = observation['bbox']
        print(f"📦 Raw Bounding Box Data: shape={bbox_data.shape if hasattr(bbox_data, 'shape') else 'No shape'}")
        if hasattr(bbox_data, 'shape') and len(bbox_data.shape) > 0:
            print(f"   Values: {bbox_data.flatten()[:12]}...")  # 显示前12个值
    
    # Keep observation as is, processing happens in the model
    return observation


def get_model(usr_args):
    """Load and return ManiBox SimpleBBoxDiffusion model"""
    print(f"🚀 Initializing ManiBox SimpleBBoxDiffusion model...")
    model = ManiBoxDiffusionModel(usr_args)
    return model


def eval(TASK_ENV, model, observation):
    """
    Evaluation function for ManiBox SimpleBBoxDiffusion policy
    """
    try:
        # Encode observation
        obs = encode_obs(observation)
        
        # Print bounding box information if available
        if 'observation' in obs and 'bbox' in obs['observation']:
            bbox_data = obs['observation']['bbox']
            print(f"📦 Bounding Box Data: shape={bbox_data.shape if hasattr(bbox_data, 'shape') else 'No shape'}")
            if hasattr(bbox_data, 'shape') and len(bbox_data.shape) > 0:
                print(f"   Values: {bbox_data.flatten()[:12]}...")  # 显示前12个值
        elif 'bbox' in obs:
            bbox_data = obs['bbox']
            print(f"📦 Bounding Box Data: shape={bbox_data.shape if hasattr(bbox_data, 'shape') else 'No shape'}")
            if hasattr(bbox_data, 'shape') and len(bbox_data.shape) > 0:
                print(f"   Values: {bbox_data.flatten()[:12]}...")  # 显示前12个值
        
        # Process observation
        processed_obs = model.encode_observation(obs)
        
        # Update observation cache (this maintains state continuity)
        model.update_obs(processed_obs)
        
        # Get single action from model (not multiple actions)
        action = model.get_action()
        
        # Execute single action (not loop through multiple actions)
        if action is not None and len(action) > 0:
            # Print the actual robot action
            action_np = action[0] if isinstance(action[0], np.ndarray) else np.array(action[0])
            print(f"🤖 Robot Action: {action_np}")
            
            # Log action to file
            if hasattr(model, 'action_log_file') and model.action_log_file is not None:
                import json
                import time
                action_log_entry = {
                    'timestep': model.timestep,
                    'action': action_np.tolist(),
                    'timestamp': time.time()
                }
                model.action_log_file.write(json.dumps(action_log_entry) + '\n')
                model.action_log_file.flush()
                model.timestep += 1
            
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
        
        return True
                
    except Exception as e:
        print(f"❌ Error in eval function: {e}")
        import traceback
        traceback.print_exc()
        # Return a safe default action to prevent complete failure
        return False


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
    
    # Reset timestep for action logging
    model.timestep = 0
    
    # Reset model state using the actual model's reset method
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
        'objects': ['bottle']
    }
    
    try:
        model = get_model(test_args)
        print("✅ Model loading test successful!")
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
