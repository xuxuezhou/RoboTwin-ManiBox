#!/usr/bin/env python3
"""
Data Reconstruction with Bounding Box Visualization

This script reconstructs robot demonstrations using ground truth actions
and saves videos with bounding box visualizations.
"""

import os
import sys
import numpy as np
import torch
import pickle
import cv2
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import yaml
import importlib
from collections import deque

# Add paths
sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError


class DataReconstructor:
    """Reconstruct robot demonstrations using ground truth actions"""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "./reconstruction_output"):
        self.config = config
        self.output_dir = output_dir
        self.episode_data = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize environment
        self.env = self._init_environment()
        
        # Initialize YOLO processor for bbox visualization
        self.yolo_processor = self._init_yolo_processor()
        
        # Video recording setup
        self.video_writers = {}
        self.video_frames = {
            'cam_high': [],
            'cam_left_wrist': [],
            'cam_right_wrist': []
        }
        
    def _init_environment(self):
        """Initialize the robot environment"""
        try:
            task_name = self.config['task_name']
            envs_module = importlib.import_module(f"envs.{task_name}")
            env_class = getattr(envs_module, task_name)
            env_instance = env_class()
            print(f"✅ Environment initialized: {task_name}")
            return env_instance
        except Exception as e:
            print(f"❌ Failed to initialize environment: {e}")
            raise
    
    def _init_yolo_processor(self):
        """Initialize YOLO processor for bounding box detection"""
        try:
            from policy.ManiBox.manibox.ManiBox.yolo_process_data import YoloProcessDataByTimeStep
            
            # Get objects from config
            objects = self.config.get('objects', ['bottle'])
            max_detections = self.config.get('max_detections_per_object', 2)
            
            # Initialize with objects from config
            yolo_processor = YoloProcessDataByTimeStep(
                objects_names=objects,
                max_detections_per_object=max_detections
            )
            print("✅ YOLO processor initialized")
            return yolo_processor
        except Exception as e:
            print(f"❌ Failed to initialize YOLO processor: {e}")
            return None
    
    def load_data(self, episode_id: int = 0):
        """Load episode data from integration.pkl"""
        try:
            # Get data path from config or use default
            data_path = self.config.get('data_path', 'policy/ManiBox/processed_data/manibox-pick-diverse-bottles')
            data_file = os.path.join(data_path, "integration.pkl")
            
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Data file not found: {data_file}")
            
            # Load data
            self.episode_data = torch.load(data_file, map_location='cpu')
            
            # Extract episode data
            episode_data = {
                'qpos': self.episode_data['qpos_data'][episode_id].numpy(),
                'action': self.episode_data['action_data'][episode_id].numpy(),
                'bbox': self.episode_data['image_data'][episode_id].numpy()
            }
            
            print(f"✅ Loaded episode {episode_id}")
            print(f"   QPOS shape: {episode_data['qpos'].shape}")
            print(f"   Action shape: {episode_data['action'].shape}")
            print(f"   BBox shape: {episode_data['bbox'].shape}")
            
            return episode_data
            
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            raise
    
    def setup_environment(self, episode_id: int = 0, seed: int = 0):
        """Setup environment for reconstruction using config"""
        try:
            # Get robot configuration from config
            def get_embodiment_file(embodiment_type):
                embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
                with open(embodiment_config_path, "r") as f:
                    _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
                
                robot_file = _embodiment_types[embodiment_type]["file_path"]
                if robot_file is None:
                    raise Exception("No embodiment files")
                return robot_file
            
            def get_embodiment_config(robot_file):
                robot_config_file = os.path.join(robot_file, "config.yml")
                with open(robot_config_file, "r", encoding="utf-8") as f:
                    embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
                return embodiment_args
            
            # Get embodiment type from config or use default
            embodiment_type = self.config.get('embodiment_type', ['aloha-agilex'])
            
            # For aloha-agilex, it's a dual-arm robot, so use the same config for both arms
            robot_file = get_embodiment_file(embodiment_type[0])
            embodiment_config = get_embodiment_config(robot_file)
            
            # Use the same config for both arms
            left_robot_file = robot_file
            right_robot_file = robot_file
            left_embodiment_config = embodiment_config
            right_embodiment_config = embodiment_config
            
            # Get camera config
            camera_config = self.config.get('camera', {
                'head_camera_type': 'D435',
                'wrist_camera_type': 'D435',
                'collect_head_camera': True,
                'collect_wrist_camera': True
            })
            
            # Get domain randomization config
            domain_randomization = self.config.get('domain_randomization', {
                'random_background': False,
                'cluttered_table': False,
                'clean_background_rate': 1,
                'random_head_camera_dis': 0,
                'random_table_height': 0,
                'random_light': False,
                'crazy_random_light_rate': 0,
                'random_embodiment': False
            })
            
            # Setup environment with config parameters
            setup_params = {
                'now_ep_num': episode_id,
                'seed': seed,
                'is_test': True,
                'domain_randomization': domain_randomization,
                'eval_mode': True,
                'eval_video_save_dir': None,
                'save_freq': -1,
                'need_plan': False,
                'left_joint_path': [],
                'right_joint_path': [],
                'left_robot_file': left_robot_file,
                'right_robot_file': right_robot_file,
                'left_embodiment_config': left_embodiment_config,
                'right_embodiment_config': right_embodiment_config,
                'embodiment_dis': 0.0,
                'dual_arm_embodied': True,  # aloha-agilex is a dual-arm robot
                'camera': camera_config
            }
            
            # Temporarily disable stability check
            original_check_stable = self.env.check_stable
            self.env.check_stable = lambda: (True, [])
            
            try:
                self.env.setup_demo(**setup_params)
                print(f"✅ Environment setup complete for episode {episode_id}")
            finally:
                # Restore original check_stable method
                self.env.check_stable = original_check_stable
                
        except Exception as e:
            print(f"❌ Failed to setup environment: {e}")
            raise
    
    def get_observation(self):
        """Get current observation from environment"""
        try:
            # Set data_type to enable RGB observation
            if not hasattr(self.env, 'data_type'):
                self.env.data_type = {
                    'rgb': True,
                    'depth': False,
                    'pointcloud': False,
                    'observer': False,
                    'endpose': False,
                    'qpos': True,
                    'mesh_segmentation': False,
                    'actor_segmentation': False
                }
            
            obs = self.env.get_obs()
            if obs is None:
                # Create a dummy observation structure
                obs = {
                    'observation': {
                        'cam_high': {'rgb': np.zeros((480, 640, 3), dtype=np.uint8)},
                        'cam_left_wrist': {'rgb': np.zeros((480, 640, 3), dtype=np.uint8)},
                        'cam_right_wrist': {'rgb': np.zeros((480, 640, 3), dtype=np.uint8)}
                    }
                }
            return obs
        except Exception as e:
            print(f"❌ Failed to get observation: {e}")
            # Return dummy observation
            return {
                'observation': {
                    'cam_high': {'rgb': np.zeros((480, 640, 3), dtype=np.uint8)},
                    'cam_left_wrist': {'rgb': np.zeros((480, 640, 3), dtype=np.uint8)},
                    'cam_right_wrist': {'rgb': np.zeros((480, 640, 3), dtype=np.uint8)}
                }
            }
    
    def take_action(self, action):
        """Take action in environment"""
        try:
            # For aloha-agilex robot, we need to handle dual arm actions
            # The action should be 14 dimensions (7 for left arm + 7 for right arm)
            if len(action) == 14:
                # Split action into left and right arm actions
                left_action = action[:7]
                right_action = action[7:]
                
                # Take actions for both arms using the environment's methods
                # Convert to pose format [x, y, z, qx, qy, qz, qw]
                left_pose = np.array(left_action)
                right_pose = np.array(right_action)
                
                self.env.left_move_to_pose(left_pose, save_freq=-1)
                self.env.right_move_to_pose(right_pose, save_freq=-1)
            else:
                # Fallback: use the first 14 dimensions
                if len(action) > 14:
                    action = action[:14]
                elif len(action) < 14:
                    action = np.pad(action, (0, 14 - len(action)), 'constant')
                
                self.env.take_action(action, action_type='qpos')
        except Exception as e:
            print(f"❌ Failed to take action: {e}")
            # Try alternative approach
            try:
                self.env.take_action(action, action_type='qpos')
            except Exception as e2:
                print(f"❌ Alternative action failed: {e2}")
    
    def process_bbox_data(self, bbox_data, timestep):
        """Process bounding box data for visualization"""
        try:
            # Reshape bbox data: (24,) -> (3, 2, 4) for 3 cameras, 2 objects, 4 coordinates
            bbox_reshaped = bbox_data.reshape(3, 2, 4)
            
            # Get images from environment
            obs = self.get_observation()
            if obs is None:
                return None
            
            images = {}
            for i, cam_name in enumerate(['cam_high', 'cam_left_wrist', 'cam_right_wrist']):
                if cam_name in obs['observation']:
                    images[cam_name] = obs['observation'][cam_name]['rgb']
            
            # Use ground truth bbox data for visualization
            bbox_for_viz = bbox_reshaped
            
            return images, bbox_for_viz
            
        except Exception as e:
            print(f"❌ Failed to process bbox data: {e}")
            return None
    
    def draw_bounding_boxes(self, image, bboxes, camera_name):
        """Draw bounding boxes on image"""
        try:
            # Convert image to uint8 if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Draw bounding boxes
            for obj_idx, bbox in enumerate(bboxes):
                if np.any(bbox != 0):  # Skip empty detections
                    x1, y1, x2, y2 = bbox
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1_px = int(x1 * width)
                    y1_px = int(y1 * height)
                    x2_px = int(x2 * width)
                    y2_px = int(y2 * height)
                    
                    # Draw rectangle
                    cv2.rectangle(image, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"Bottle_{obj_idx}"
                    cv2.putText(image, label, (x1_px, y1_px - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return image
            
        except Exception as e:
            print(f"❌ Failed to draw bounding boxes: {e}")
            return image
    
    def save_frame(self, images, bboxes, timestep):
        """Save frame with bounding boxes"""
        try:
            for i, cam_name in enumerate(['cam_high', 'cam_left_wrist', 'cam_right_wrist']):
                if cam_name in images:
                    image = images[cam_name].copy()
                    
                    # Draw bounding boxes
                    if bboxes is not None and i < len(bboxes):
                        image = self.draw_bounding_boxes(image, bboxes[i], cam_name)
                    
                    # Add timestep info
                    cv2.putText(image, f"Step: {timestep}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Store frame
                    self.video_frames[cam_name].append(image)
                    
        except Exception as e:
            print(f"❌ Failed to save frame: {e}")
    
    def create_video(self, episode_id: int, fps: int = 10):
        """Create video from stored frames"""
        try:
            for cam_name, frames in self.video_frames.items():
                if not frames:
                    continue
                
                # Get video dimensions
                height, width = frames[0].shape[:2]
                
                # Create video writer with compatible codec
                video_path = os.path.join(self.output_dir, f"episode_{episode_id}_{cam_name}_bbox.mp4")
                
                # Use XVID codec which is more widely supported
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    # Fallback: save as image sequence and use ffmpeg
                    print(f"⚠️ Video codec failed, saving as image sequence for {cam_name}")
                    temp_dir = os.path.join(self.output_dir, f"temp_{cam_name}")
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    for i, frame in enumerate(frames):
                        img_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                        cv2.imwrite(img_path, frame)
                    
                    # Use ffmpeg to create video
                    try:
                        import subprocess
                        cmd = [
                            'ffmpeg', '-y', '-framerate', str(fps),
                            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                            video_path
                        ]
                        subprocess.run(cmd, check=True, capture_output=True)
                        print(f"✅ Video saved with ffmpeg: {video_path}")
                        
                        # Clean up temp directory
                        import shutil
                        shutil.rmtree(temp_dir)
                    except Exception as ffmpeg_error:
                        print(f"⚠️ FFmpeg failed: {ffmpeg_error}")
                        continue
                else:
                    # Write frames
                    for frame in frames:
                        out.write(frame)
                    
                    out.release()
                    print(f"✅ Video saved: {video_path}")
                
        except Exception as e:
            print(f"❌ Failed to create video: {e}")
            # Fallback: save as image sequence
            for cam_name, frames in self.video_frames.items():
                if frames:
                    print(f"⚠️ Saving {cam_name} as image sequence due to video error")
                    for i, frame in enumerate(frames):
                        img_path = os.path.join(self.output_dir, f"episode_{episode_id}_{cam_name}_frame_{i:04d}.png")
                        cv2.imwrite(img_path, frame)
    
    def reconstruct_episode(self, episode_id: int = 0, seed: int = 0, max_steps: int = None):
        """Reconstruct a single episode"""
        try:
            print(f"🔄 Starting reconstruction of episode {episode_id}")
            
            # Load episode data
            episode_data = self.load_data(episode_id)
            
            # Setup environment
            self.setup_environment(episode_id, seed)
            
            # Get episode length
            episode_length = len(episode_data['action'])
            if max_steps is not None:
                episode_length = min(episode_length, max_steps)
            
            print(f"📊 Episode length: {episode_length} steps")
            
            # Clear video frames
            for cam_name in self.video_frames:
                self.video_frames[cam_name].clear()
            
            # Reconstruct episode step by step
            for timestep in range(episode_length):
                print(f"   Step {timestep + 1}/{episode_length}")
                
                # Get current action
                action = episode_data['action'][timestep]
                
                # Get current bbox data
                bbox_data = episode_data['bbox'][timestep]
                
                # Process bbox data and get images
                result = self.process_bbox_data(bbox_data, timestep)
                if result is not None:
                    images, bboxes = result
                    
                    # Save frame with bounding boxes
                    self.save_frame(images, bboxes, timestep)
                
                # Take action
                self.take_action(action)
                
                # Small delay for visualization
                time.sleep(0.01)
            
            # Create video
            self.create_video(episode_id)
            
            # Close environment
            self.env.close_env()
            
            print(f"✅ Episode {episode_id} reconstruction completed")
            
        except Exception as e:
            print(f"❌ Failed to reconstruct episode: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'env'):
                self.env.close_env()


def parse_args_and_config():
    """Parse arguments and load config file"""
    parser = argparse.ArgumentParser(description='Reconstruct robot demonstrations with bounding box visualization')
    parser.add_argument("--config", type=str, default="policy/ManiBox/deploy_policy_diffusion.yml",
                       help='Path to config file')
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    parser.add_argument('--episode-id', type=int, default=0,
                       help='Episode ID to reconstruct (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Maximum number of steps to reconstruct (default: all)')
    parser.add_argument('--output-dir', type=str, default='./reconstruction_output',
                       help='Output directory for videos (default: ./reconstruction_output)')
    parser.add_argument('--data-path', type=str, 
                       default='policy/ManiBox/processed_data/manibox-pick-diverse-bottles',
                       help='Path to processed data directory')
    
    args = parser.parse_args()
    
    # Load config file
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict
    
    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)
    
    # Add command line arguments to config
    config['episode_id'] = args.episode_id
    config['seed'] = args.seed
    config['max_steps'] = args.max_steps
    config['output_dir'] = args.output_dir
    config['data_path'] = args.data_path
    
    return config


def main():
    try:
        # Parse config
        config = parse_args_and_config()
        
        # Create reconstructor
        reconstructor = DataReconstructor(
            config=config,
            output_dir=config['output_dir']
        )
        
        # Reconstruct episode
        reconstructor.reconstruct_episode(
            episode_id=config['episode_id'],
            seed=config['seed'],
            max_steps=config['max_steps']
        )
        
        print(f"🎉 Reconstruction completed! Videos saved in: {config['output_dir']}")
        
    except Exception as e:
        print(f"❌ Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 