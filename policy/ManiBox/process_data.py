#!/usr/bin/env python3
"""
ManiBox Data Processor
Converts ACT format HDF5 data to ManiBox integration.pkl format
Only requires basic dependencies: torch, h5py, numpy, opencv, ultralytics
"""

import sys
import os
import h5py
import numpy as np
import cv2
import argparse
import json
import torch
from tqdm import tqdm

# YOLO for object detection
os.environ['YOLO_VERBOSE'] = str(False)
from ultralytics import YOLO


class ManiBoxDataProcessor:
    def __init__(self, objects_names=["apple"], batch_size=8, use_cpu=False):
        """Initialize ManiBox data processor with YOLO detection"""
        self.objects_names = objects_names
        self.camera_names = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        self.batch_size = batch_size
        self.use_cpu = use_cpu
        
        # Initialize YOLO model
        device = 'cpu' if use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.detection_model = YOLO("yolov8l-world.pt")
        self.detection_model.set_classes(self.objects_names)
        
        # Move model to specified device
        if hasattr(self.detection_model.model, 'to'):
            self.detection_model.model.to(device)
        
        print(f"🔧 Using device: {device}")
        print(f"📦 Batch size: {batch_size}")
        
        # Camera mapping from ACT to ManiBox format
        self.camera_mapping = {
            "head_camera": "cam_high",
            "right_camera": "cam_right_wrist", 
            "left_camera": "cam_left_wrist"
        }
        
    def load_act_episode(self, dataset_path):
        """Load single episode from ACT format HDF5"""
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")

        with h5py.File(dataset_path, "r") as root:
            # Load joint actions
            left_gripper = root["/joint_action/left_gripper"][()]
            left_arm = root["/joint_action/left_arm"][()]
            right_gripper = root["/joint_action/right_gripper"][()]
            right_arm = root["/joint_action/right_arm"][()]
            
            # Load images
            image_dict = {}
            for cam_name in root["/observation/"].keys():
                image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

        return left_gripper, left_arm, right_gripper, right_arm, image_dict

    def detect_objects_in_images(self, images):
        """Detect objects in batch of images using YOLO with memory management"""
        if len(images) == 0:
            return torch.zeros(0, len(self.objects_names) * 4)
            
        # Convert images for YOLO processing
        processed_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                # Convert BGR to RGB for YOLO
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed_images.append(img_rgb)
        
        # Process images in smaller batches to avoid memory issues
        all_batch_bboxes = []
        
        for i in range(0, len(processed_images), self.batch_size):
            batch_images = processed_images[i:i + self.batch_size]
            
            # Clear GPU cache before processing
            if torch.cuda.is_available() and not self.use_cpu:
                torch.cuda.empty_cache()
            
            try:
                # Batch detection with memory management
                with torch.no_grad():
                    results = self.detection_model.predict(
                        batch_images, 
                        verbose=False,
                        device='cpu' if self.use_cpu else None
                    )
                
                # Extract bounding boxes for this batch
                for result in results:
                    bbox_list = [0, 0, 0, 0] * len(self.objects_names)  # Default no detection
                    
                    if result.boxes is not None:
                        for box in result.boxes:
                            # Get normalized coordinates [x1, y1, x2, y2]
                            bbox = box.xyxyn.squeeze().cpu().numpy().tolist()
                            class_name = result.names[int(box.cls.item())]
                            
                            if class_name in self.objects_names:
                                obj_idx = self.objects_names.index(class_name)
                                start_idx = obj_idx * 4
                                bbox_list[start_idx:start_idx+4] = bbox
                                
                    all_batch_bboxes.append(bbox_list)
                    
                # Clear memory after each batch
                del results
                if torch.cuda.is_available() and not self.use_cpu:
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️  GPU memory error at batch {i//self.batch_size + 1}. Switching to CPU...")
                # Fallback to CPU for this batch
                with torch.no_grad():
                    results = self.detection_model.predict(batch_images, verbose=False, device='cpu')
                
                for result in results:
                    bbox_list = [0, 0, 0, 0] * len(self.objects_names)
                    
                    if result.boxes is not None:
                        for box in result.boxes:
                            bbox = box.xyxyn.squeeze().cpu().numpy().tolist()
                            class_name = result.names[int(box.cls.item())]
                            
                            if class_name in self.objects_names:
                                obj_idx = self.objects_names.index(class_name)
                                start_idx = obj_idx * 4
                                bbox_list[start_idx:start_idx+4] = bbox
                                
                    all_batch_bboxes.append(bbox_list)
                
                del results
        
        return torch.tensor(all_batch_bboxes, dtype=torch.float32)

    def process_episode(self, episode_path):
        """Process single episode to extract qpos, actions, and bbox features"""
        # Load ACT data
        left_gripper_all, left_arm_all, right_gripper_all, right_arm_all, image_dict = self.load_act_episode(episode_path)
        
        episode_len = left_gripper_all.shape[0] - 1  # Exclude last timestep for action
        
        # Process states and actions
        qpos_data = []
        action_data = []
        
        # Collect images from all cameras and timesteps
        cam_images = {cam_name: [] for cam_name in self.camera_names}
        
        for j in range(episode_len):
            # Create current state (qpos)
            left_gripper, left_arm = left_gripper_all[j], left_arm_all[j]
            right_gripper, right_arm = right_gripper_all[j], right_arm_all[j]
            state = np.concatenate((left_arm, [left_gripper], right_arm, [right_gripper]), axis=0)
            qpos_data.append(state.astype(np.float32))
            
            # Create next action
            if j < episode_len - 1:
                next_left_gripper, next_left_arm = left_gripper_all[j + 1], left_arm_all[j + 1]
                next_right_gripper, next_right_arm = right_gripper_all[j + 1], right_arm_all[j + 1]
                next_state = np.concatenate((next_left_arm, [next_left_gripper], next_right_arm, [next_right_gripper]), axis=0)
            else:
                next_state = state  # Use current state for last timestep
            action_data.append(next_state.astype(np.float32))
            
            # Decode and collect images
            for act_cam_name, our_cam_name in self.camera_mapping.items():
                if act_cam_name in image_dict and j < len(image_dict[act_cam_name]):
                    camera_bits = image_dict[act_cam_name][j]
                    image = cv2.imdecode(np.frombuffer(camera_bits, np.uint8), cv2.IMREAD_COLOR)
                    cam_images[our_cam_name].append(image)
        
        # Process all images with YOLO to get bounding boxes
        all_bboxes = []
        for cam_name in self.camera_names:
            if cam_name in cam_images:
                cam_bboxes = self.detect_objects_in_images(cam_images[cam_name])
                all_bboxes.append(cam_bboxes)
        
        # Concatenate bboxes from all cameras: (episode_len, num_cameras * num_objects * 4)
        if all_bboxes:
            bbox_data = torch.cat(all_bboxes, dim=1)
        else:
            bbox_data = torch.zeros(episode_len, len(self.camera_names) * len(self.objects_names) * 4)
        
        return np.array(qpos_data), np.array(action_data), bbox_data

    def process_dataset(self, data_path, num_episodes, save_path):
        """Process multiple episodes and save as integration.pkl"""
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Process episodes
        all_qpos_data = []
        all_action_data = []
        all_image_data = []
        
        print(f"Processing {num_episodes} episodes for ManiBox format...")
        
        for i in tqdm(range(num_episodes)):
            episode_path = os.path.join(data_path, f"episode{i}.hdf5")
            
            if not os.path.exists(episode_path):
                print(f"Warning: Episode {i} not found, skipping...")
                continue
                
            qpos_data, action_data, bbox_data = self.process_episode(episode_path)
            
            all_qpos_data.append(qpos_data)
            all_action_data.append(action_data)
            all_image_data.append(bbox_data)
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{num_episodes} episodes")
        
        # Convert to tensors and stack
        print("Converting to tensors and saving...")
        
        # Find the minimum episode length to ensure consistency
        min_episode_len = min(len(qpos) for qpos in all_qpos_data)
        
        # Truncate all episodes to minimum length for consistency
        qpos_tensor = torch.stack([torch.from_numpy(qpos[:min_episode_len]) for qpos in all_qpos_data])
        action_tensor = torch.stack([torch.from_numpy(action[:min_episode_len]) for action in all_action_data])
        image_tensor = torch.stack([bbox[:min_episode_len] for bbox in all_image_data])
        
        # Create data dictionary in ManiBox format
        data = {
            "image_data": image_tensor.float(),      # (num_episodes, episode_len, bbox_dim)
            "qpos_data": qpos_tensor.float(),        # (num_episodes, episode_len, 14)
            "action_data": action_tensor.float(),    # (num_episodes, episode_len, 14)
        }
        
        # Save as integration.pkl
        integration_path = os.path.join(save_path, "integration.pkl")
        torch.save(data, integration_path)
        
        print(f"✅ Successfully saved {len(all_qpos_data)} episodes to {integration_path}")
        print(f"📊 Data shapes:")
        print(f"   - image_data: {data['image_data'].shape}")
        print(f"   - qpos_data: {data['qpos_data'].shape}")
        print(f"   - action_data: {data['action_data'].shape}")
        
        return len(all_qpos_data)


def main():
    parser = argparse.ArgumentParser(description="ManiBox Data Processor: Convert ACT to ManiBox format")
    parser.add_argument("task_name", type=str, help="Task name (e.g., grasp_apple)")
    parser.add_argument("task_config", type=str, help="Task configuration")
    parser.add_argument("num_episodes", type=int, help="Number of episodes to process")
    parser.add_argument("--objects", type=str, default="apple", 
                        help="Objects to detect (comma-separated)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Custom data path (default: ../../data/task_name/task_config/data)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Custom save path (default: processed_data/manibox-task_name)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for YOLO processing (default: 4, reduce if GPU memory issues)")
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Force CPU processing (slower but no GPU memory issues)")
    
    args = parser.parse_args()
    
    # Parse objects
    objects_names = [obj.strip() for obj in args.objects.split(",")]
    
    # Set paths
    if args.data_path is None:
        data_path = os.path.join("../../data", args.task_name, args.task_config, "data")
    else:
        data_path = args.data_path
        
    if args.save_path is None:
        save_path = f"processed_data/manibox-{args.task_name}"
    else:
        save_path = args.save_path
    
    print(f"🚀 ManiBox Data Processor")
    print(f"📁 Input: {data_path}")
    print(f"💾 Output: {save_path}")
    print(f"🎯 Objects: {objects_names}")
    print(f"📈 Episodes: {args.num_episodes}")
    if args.cpu:
        print(f"⚙️  Processing mode: CPU (slower but safe)")
    else:
        print(f"⚙️  Processing mode: GPU with batch size {args.batch_size}")
    print("-" * 50)
    
    # Check GPU memory if using GPU
    if not args.cpu and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔍 GPU Memory: {gpu_memory:.1f} GB")
        if gpu_memory < 8:
            print(f"⚠️  Warning: GPU memory < 8GB, consider using --cpu or reducing --batch_size")
    
    # Initialize processor and run
    processor = ManiBoxDataProcessor(
        objects_names=objects_names, 
        batch_size=args.batch_size,
        use_cpu=args.cpu
    )
    processed_count = processor.process_dataset(data_path, args.num_episodes, save_path)
    
    print(f"🎉 Processing completed! Processed {processed_count} episodes.")


if __name__ == "__main__":
    main() 