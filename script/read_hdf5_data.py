#!/usr/bin/env python3
"""
HDF5 Data Reader Script for RoboTwin

This script provides utilities to read and analyze HDF5 files containing robot demonstration data.
It can display file structure, extract specific data, and provide basic statistics.

Usage:
    python read_hdf5_data.py --file path/to/episode_0.hdf5
    python read_hdf5_data.py --dir path/to/data/directory --episode 0
    python read_hdf5_data.py --dir path/to/data/directory --list
"""

import os
import sys
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, Any, List, Optional, Tuple


class HDF5DataReader:
    """HDF5 data reader for robot demonstration files"""
    
    def __init__(self, file_path: str):
        """Initialize the reader with a file path"""
        self.file_path = file_path
        self.data = None
        
    def load_data(self) -> Dict[str, Any]:
        """Load all data from the HDF5 file"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        data = {}
        with h5py.File(self.file_path, 'r') as f:
            # Load attributes
            data['attributes'] = dict(f.attrs)
            
            # Load main datasets
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data[key] = f[key][()]
                elif isinstance(f[key], h5py.Group):
                    data[key] = self._load_group(f[key])
                    
        self.data = data
        return data
    
    def _load_group(self, group: h5py.Group) -> Dict[str, Any]:
        """Recursively load a group and its contents"""
        group_data = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                group_data[key] = group[key][()]
            elif isinstance(group[key], h5py.Group):
                group_data[key] = self._load_group(group[key])
        return group_data
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get basic file information"""
        if self.data is None:
            self.load_data()
            
        info = {
            'file_path': self.file_path,
            'file_size_mb': os.path.getsize(self.file_path) / (1024 * 1024),
            'attributes': self.data.get('attributes', {}),
            'datasets': {},
            'groups': {}
        }
        
        # Analyze datasets
        for key, value in self.data.items():
            if key == 'attributes':
                continue
            elif isinstance(value, dict):
                info['groups'][key] = self._analyze_group(value)
            else:
                info['datasets'][key] = self._analyze_dataset(value)
                
        return info
    
    def _analyze_dataset(self, dataset: np.ndarray) -> Dict[str, Any]:
        """Analyze a dataset and return basic statistics"""
        if dataset is None:
            return {'type': 'None', 'shape': None}
            
        analysis = {
            'type': str(dataset.dtype),
            'shape': dataset.shape,
            'size': dataset.size,
            'memory_mb': dataset.nbytes / (1024 * 1024)
        }
        
        if dataset.size > 0:
            if np.issubdtype(dataset.dtype, np.number):
                analysis.update({
                    'min': float(np.min(dataset)),
                    'max': float(np.max(dataset)),
                    'mean': float(np.mean(dataset)),
                    'std': float(np.std(dataset))
                })
                
        return analysis
    
    def _analyze_group(self, group_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a group and return summary"""
        analysis = {
            'datasets': {},
            'subgroups': {}
        }
        
        for key, value in group_data.items():
            if isinstance(value, dict):
                analysis['subgroups'][key] = self._analyze_group(value)
            else:
                analysis['datasets'][key] = self._analyze_dataset(value)
                
        return analysis
    
    def get_episode_length(self) -> int:
        """Get the length of the episode (number of timesteps)"""
        if self.data is None:
            self.load_data()
            
        # Try different possible keys for episode length
        possible_keys = ['action', 'observations/qpos', 'observations/qvel', 'joint_action/vector']
        for key in possible_keys:
            if key in self.data:
                return len(self.data[key])
            elif 'observations' in self.data and '/' in key and key.split('/')[1] in self.data['observations']:
                return len(self.data['observations'][key.split('/')[1]])
            elif 'joint_action' in self.data and '/' in key and key.split('/')[1] in self.data['joint_action']:
                return len(self.data['joint_action'][key.split('/')[1]])
                
        return 0
    
    def get_robot_state(self, timestep: int = 0) -> Dict[str, Any]:
        """Get robot state at a specific timestep"""
        if self.data is None:
            self.load_data()
            
        state = {}
        
        # Get qpos (joint positions)
        if 'observations' in self.data and 'qpos' in self.data['observations']:
            qpos = self.data['observations']['qpos']
            if timestep < len(qpos):
                state['qpos'] = qpos[timestep]
                
        # Get qvel (joint velocities)
        if 'observations' in self.data and 'qvel' in self.data['observations']:
            qvel = self.data['observations']['qvel']
            if timestep < len(qvel):
                state['qvel'] = qvel[timestep]
                
        # Get effort (joint efforts)
        if 'observations' in self.data and 'effort' in self.data['observations']:
            effort = self.data['observations']['effort']
            if timestep < len(effort):
                state['effort'] = effort[timestep]
                
        # Get joint actions (alternative format)
        if 'joint_action' in self.data:
            joint_action = self.data['joint_action']
            if 'vector' in joint_action and timestep < len(joint_action['vector']):
                state['joint_action'] = joint_action['vector'][timestep]
            if 'left_arm' in joint_action and timestep < len(joint_action['left_arm']):
                state['left_arm'] = joint_action['left_arm'][timestep]
            if 'right_arm' in joint_action and timestep < len(joint_action['right_arm']):
                state['right_arm'] = joint_action['right_arm'][timestep]
            if 'left_gripper' in joint_action and timestep < len(joint_action['left_gripper']):
                state['left_gripper'] = joint_action['left_gripper'][timestep]
            if 'right_gripper' in joint_action and timestep < len(joint_action['right_gripper']):
                state['right_gripper'] = joint_action['right_gripper'][timestep]
                
        return state
    
    def get_action(self, timestep: int = 0) -> Optional[np.ndarray]:
        """Get action at a specific timestep"""
        if self.data is None:
            self.load_data()
            
        if 'action' in self.data and timestep < len(self.data['action']):
            return self.data['action'][timestep]
            
        # Try joint_action format
        if 'joint_action' in self.data and 'vector' in self.data['joint_action']:
            joint_action = self.data['joint_action']['vector']
            if timestep < len(joint_action):
                return joint_action[timestep]
                
        return None
    
    def get_images(self, timestep: int = 0) -> Dict[str, np.ndarray]:
        """Get images from all cameras at a specific timestep"""
        if self.data is None:
            self.load_data()
            
        images = {}
        
        if 'observations' in self.data and 'images' in self.data['observations']:
            image_data = self.data['observations']['images']
            for camera_name, camera_data in image_data.items():
                if timestep < len(camera_data):
                    images[camera_name] = camera_data[timestep]
                    
        return images
    
    def get_depth_images(self, timestep: int = 0) -> Dict[str, np.ndarray]:
        """Get depth images from all cameras at a specific timestep"""
        if self.data is None:
            self.load_data()
            
        depth_images = {}
        
        if 'observations' in self.data and 'images_depth' in self.data['observations']:
            depth_data = self.data['observations']['images_depth']
            for camera_name, camera_data in depth_data.items():
                if timestep < len(camera_data):
                    depth_images[camera_name] = camera_data[timestep]
                    
        return depth_images
    
    def plot_episode_summary(self, save_path: Optional[str] = None):
        """Create a summary plot of the episode"""
        if self.data is None:
            self.load_data()
            
        episode_length = self.get_episode_length()
        if episode_length == 0:
            print("No data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Episode Summary: {os.path.basename(self.file_path)}')
        
        # Plot joint positions
        if 'observations' in self.data and 'qpos' in self.data['observations']:
            qpos = self.data['observations']['qpos']
            axes[0, 0].plot(qpos)
            axes[0, 0].set_title('Joint Positions (qpos)')
            axes[0, 0].set_xlabel('Timestep')
            axes[0, 0].set_ylabel('Position')
            axes[0, 0].grid(True)
            
        # Plot joint velocities
        if 'observations' in self.data and 'qvel' in self.data['observations']:
            qvel = self.data['observations']['qvel']
            axes[0, 1].plot(qvel)
            axes[0, 1].set_title('Joint Velocities (qvel)')
            axes[0, 1].set_xlabel('Timestep')
            axes[0, 1].set_ylabel('Velocity')
            axes[0, 1].grid(True)
            
        # Plot actions
        if 'action' in self.data:
            action = self.data['action']
            axes[1, 0].plot(action)
            axes[1, 0].set_title('Actions')
            axes[1, 0].set_xlabel('Timestep')
            axes[1, 0].set_ylabel('Action')
            axes[1, 0].grid(True)
            
        # Plot base actions if available
        if 'base_action' in self.data:
            base_action = self.data['base_action']
            axes[1, 1].plot(base_action)
            axes[1, 1].set_title('Base Actions')
            axes[1, 1].set_xlabel('Timestep')
            axes[1, 1].set_ylabel('Base Action')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No base action data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Base Actions')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
            
    def save_summary(self, output_path: str):
        """Save a summary of the data to a JSON file"""
        if self.data is None:
            self.load_data()
            
        summary = {
            'file_info': self.get_file_info(),
            'episode_length': self.get_episode_length(),
            'sample_data': {
                'first_timestep': {
                    'robot_state': self.get_robot_state(0),
                    'action': self.get_action(0).tolist() if self.get_action(0) is not None else None
                },
                'last_timestep': {
                    'robot_state': self.get_robot_state(-1),
                    'action': self.get_action(-1).tolist() if self.get_action(-1) is not None else None
                }
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        print(f"Summary saved to: {output_path}")


def find_hdf5_files(directory: str) -> List[str]:
    """Find all HDF5 files in a directory"""
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    return sorted(hdf5_files)


def main():
    parser = argparse.ArgumentParser(description='Read and analyze HDF5 robot demonstration data')
    parser.add_argument('--file', type=str, help='Path to a specific HDF5 file')
    parser.add_argument('--dir', type=str, help='Directory containing HDF5 files')
    parser.add_argument('--episode', type=int, default=0, help='Episode number (when using --dir)')
    parser.add_argument('--list', action='store_true', help='List all HDF5 files in directory')
    parser.add_argument('--info', action='store_true', help='Show detailed file information')
    parser.add_argument('--plot', action='store_true', help='Create summary plot')
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
    parser.add_argument('--save-summary', type=str, help='Save summary to JSON file')
    parser.add_argument('--timestep', type=int, default=0, help='Timestep to examine')
    
    args = parser.parse_args()
    
    if args.file:
        # Read specific file
        file_path = args.file
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
            
    elif args.dir:
        # Find file in directory
        if args.list:
            hdf5_files = find_hdf5_files(args.dir)
            print(f"Found {len(hdf5_files)} HDF5 files:")
            for file_path in hdf5_files:
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  {file_path} ({file_size:.2f} MB)")
            return
            
        # Look for episode file
        possible_paths = [
            os.path.join(args.dir, f'episode_{args.episode}.hdf5'),
            os.path.join(args.dir, f'data/episode_{args.episode}.hdf5'),
            os.path.join(args.dir, f'episode{args.episode}.hdf5'),
            os.path.join(args.dir, f'data/episode{args.episode}.hdf5')
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
                
        if file_path is None:
            print(f"Error: Episode {args.episode} not found in {args.dir}")
            print("Available files:")
            hdf5_files = find_hdf5_files(args.dir)
            for f in hdf5_files[:10]:  # Show first 10 files
                print(f"  {f}")
            if len(hdf5_files) > 10:
                print(f"  ... and {len(hdf5_files) - 10} more files")
            return
    else:
        parser.print_help()
        return
        
    # Create reader and load data
    print(f"Reading file: {file_path}")
    reader = HDF5DataReader(file_path)
    
    try:
        reader.load_data()
        print("✓ Data loaded successfully")
        
        # Show basic info
        info = reader.get_file_info()
        print(f"\nFile Information:")
        print(f"  Size: {info['file_size_mb']:.2f} MB")
        print(f"  Episode length: {reader.get_episode_length()} timesteps")
        print(f"  Attributes: {info['attributes']}")
        
        if args.info:
            print(f"\nDetailed Information:")
            print(json.dumps(info, indent=2, default=str))
            
        # Show timestep data
        print(f"\nTimestep {args.timestep} data:")
        robot_state = reader.get_robot_state(args.timestep)
        action = reader.get_action(args.timestep)
        
        if robot_state:
            print("  Robot State:")
            for key, value in robot_state.items():
                if isinstance(value, np.ndarray):
                    print(f"    {key}: shape={value.shape}, values={value[:5]}...")
                else:
                    print(f"    {key}: {value}")
                    
        if action is not None:
            print(f"  Action: shape={action.shape}, values={action[:5]}...")
            
        # Create plot
        if args.plot or args.save_plot:
            reader.plot_episode_summary(args.save_plot)
            
        # Save summary
        if args.save_summary:
            reader.save_summary(args.save_summary)
            
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 