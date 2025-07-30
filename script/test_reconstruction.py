#!/usr/bin/env python3
"""
Test Data Reconstruction

A simplified test script to verify data reconstruction functionality.
"""

import os
import sys
import numpy as np
import torch

# Add paths
sys.path.append("./")
sys.path.append("./policy")

def test_data_loading():
    """Test loading data from integration.pkl"""
    print("🧪 Testing data loading...")
    
    data_path = "policy/ManiBox/processed_data/manibox-pick-diverse-bottles"
    data_file = os.path.join(data_path, "integration.pkl")
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    
    try:
        # Load data
        data = torch.load(data_file, map_location='cpu')
        
        print(f"✅ Data loaded successfully")
        print(f"   Keys: {list(data.keys())}")
        
        # Check data structure
        if 'qpos_data' in data:
            print(f"   QPOS data shape: {data['qpos_data'].shape}")
        if 'action_data' in data:
            print(f"   Action data shape: {data['action_data'].shape}")
        if 'image_data' in data:
            print(f"   Image data shape: {data['image_data'].shape}")
        
        # Test episode extraction
        episode_id = 0
        episode_data = {
            'qpos': data['qpos_data'][episode_id].numpy(),
            'action': data['action_data'][episode_id].numpy(),
            'bbox': data['image_data'][episode_id].numpy()
        }
        
        print(f"✅ Episode {episode_id} extracted")
        print(f"   QPOS shape: {episode_data['qpos'].shape}")
        print(f"   Action shape: {episode_data['action'].shape}")
        print(f"   BBox shape: {episode_data['bbox'].shape}")
        
        # Test bbox reshaping
        bbox_data = episode_data['bbox'][0]  # First timestep
        print(f"   BBox data (first timestep): {bbox_data.shape}")
        
        # Reshape to (3, 2, 4) for 3 cameras, 2 objects, 4 coordinates
        bbox_reshaped = bbox_data.reshape(3, 2, 4)
        print(f"   BBox reshaped: {bbox_reshaped.shape}")
        
        # Show some bbox values
        for cam_idx in range(3):
            for obj_idx in range(2):
                bbox = bbox_reshaped[cam_idx, obj_idx]
                if np.any(bbox != 0):
                    print(f"   Camera {cam_idx}, Object {obj_idx}: {bbox}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yolo_processor():
    """Test YOLO processor initialization"""
    print("\n🧪 Testing YOLO processor...")
    
    try:
        from policy.ManiBox.manibox.ManiBox.yolo_process_data import YoloProcessDataByTimeStep
        
        # Initialize YOLO processor
        yolo_processor = YoloProcessDataByTimeStep(
            objects_names=['bottle'],
            max_detections_per_object=2
        )
        
        print("✅ YOLO processor initialized successfully")
        print(f"   Objects: {yolo_processor.objects_names}")
        print(f"   Max detections per object: {yolo_processor.max_detections_per_object}")
        print(f"   Total detections: {yolo_processor.total_detections}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize YOLO processor: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment():
    """Test environment initialization"""
    print("\n🧪 Testing environment initialization...")
    
    try:
        import importlib
        envs_module = importlib.import_module("envs.pick_diverse_bottles")
        env_class = getattr(envs_module, "pick_diverse_bottles")
        env_instance = env_class()
        
        print("✅ Environment initialized successfully")
        print(f"   Environment class: {env_class.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize environment: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Running data reconstruction tests...")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_yolo_processor,
        test_environment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Ready to run reconstruction.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 