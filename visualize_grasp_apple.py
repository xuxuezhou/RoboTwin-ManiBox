#!/usr/bin/env python3
import os
import sys
import numpy as np
import yaml
import time

# Add the current directory to Python path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from envs.grasp_apple import grasp_apple
    print("✓ Successfully imported grasp_apple")
except ImportError as e:
    print(f"✗ Failed to import grasp_apple: {e}")
    print("Make sure you're running from the RoboTwin root directory")
    sys.exit(1)

def load_embodiment_config():
    """Load the robot embodiment configuration"""
    # Load embodiment config
    embodiment_config_path = "task_config/_embodiment_config.yml"
    with open(embodiment_config_path, 'r', encoding='utf-8') as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # Use aloha-agilex as default
    embodiment = "aloha-agilex"
    robot_file = embodiment_types[embodiment]['file_path']
    
    # Load robot config
    robot_config_file = os.path.join(robot_file, 'config.yml')
    with open(robot_config_file, 'r', encoding='utf-8') as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    return robot_file, embodiment_args

def create_kwargs(robot_file, embodiment_args, seed=None):
    """Create configuration parameters for task"""
    if seed is None:
        seed = np.random.randint(0, 10000)
    
    return {
        'seed': seed,
        'render_freq': 5,  # Render every 5 steps (adjust for performance)
        'task_name': 'grasp_apple',
        'save_data': False,  # Don't save data during visualization
        'eval_mode': False,
        'dual_arm': True,
        'need_plan': True,
        'dual_arm_embodied': embodiment_args.get('dual_arm', True),
        
        # Robot configuration
        'left_embodiment_config': embodiment_args,
        'right_embodiment_config': embodiment_args,
        'left_robot_file': robot_file,
        'right_robot_file': robot_file,
        
        # Camera configuration
        'camera': {
            'head_camera_type': 'D435',
            'wrist_camera_type': 'D435',
            'collect_head_camera': True,
            'collect_wrist_camera': True
        },
        
        'domain_randomization': {
            'random_background': False,
            'cluttered_table': False,
            'clean_background_rate': 1.0,
            'random_head_camera_dis': 0,
            'random_table_height': 0,
            'random_light': False,
            'crazy_random_light_rate': 0,
            'random_embodiment': False
        },
        'data_type': {
            'rgb': True,
            'depth': False,
            'pointcloud': False,
            'endpose': True,
            'qpos': True,
            'mesh_segmentation': False,
            'actor_segmentation': False,
            'third_view': False,
            'conbine': False
        }
    }

def run_single_attempt(task, attempt_num):
    """Run a single grasp attempt"""
    print(f"\n{'='*50}")
    print(f"🍎 ATTEMPT #{attempt_num}")
    print(f"{'='*50}")
    
    try:
        # Print debug information
        print(f"Apple position: {task.apple.get_pose().p}")
        print(f"Using arm: {'right' if task.qpose_tag == 1 else 'left'}")
        print(f"Target pose: {task.right_target_pose if task.qpose_tag == 1 else task.left_target_pose}")
        
        # Execute the task
        print("🤖 Executing grasp_apple task...")
        result = task.play_once()
        print("✓ Task execution complete")
        
        # Print final apple position for debugging
        final_apple_pos = task.apple.get_pose().p
        print(f"Final apple position: {final_apple_pos}")
        print(f"Target height threshold: 0.9")
        print(f"Apple height: {final_apple_pos[2]:.3f}")
        print(f"Apple x-position: {final_apple_pos[0]:.3f}")
        print(f"Success criteria: arm={'right' if task.qpose_tag == 1 else 'left'}, x_threshold={'> 0.15' if task.qpose_tag == 1 else '< -0.15'}")
        
        # Check success
        success = task.check_success()
        if success:
            print("🎉 Task SUCCESS! Apple successfully grasped and moved!")
        else:
            print("❌ Task FAILED. Apple not at target position.")
            # Provide detailed failure reason
            if task.qpose_tag == 1:  # right arm
                if final_apple_pos[0] <= 0.15:
                    print(f"   Reason: Apple x-position ({final_apple_pos[0]:.3f}) should be > 0.15")
            else:  # left arm
                if final_apple_pos[0] >= -0.15:
                    print(f"   Reason: Apple x-position ({final_apple_pos[0]:.3f}) should be < -0.15")
            if final_apple_pos[2] <= 0.9:
                print(f"   Reason: Apple height ({final_apple_pos[2]:.3f}) should be > 0.9")
        
        return success
        
    except Exception as e:
        print(f"✗ Error during attempt {attempt_num}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def reset_task(task, robot_file, embodiment_args):
    """Reset the task environment for a new attempt"""
    try:
        # Close current environment
        task.close_env()
        
        # Create new instance with different seed for randomization
        new_seed = np.random.randint(0, 10000)
        kwargs = create_kwargs(robot_file, embodiment_args, new_seed)
        
        # Create new task instance
        new_task = grasp_apple()
        new_task.setup_demo(**kwargs)
        
        return new_task
        
    except Exception as e:
        print(f"⚠️ Error resetting task: {e}")
        return None

def main():
    print("🍎 Starting automatic grasp_apple visualization...")
    print("Running 5 attempts automatically...")
    
    # Load robot configuration
    try:
        robot_file, embodiment_args = load_embodiment_config()
        print("✓ Successfully loaded robot configuration")
    except Exception as e:
        print(f"✗ Failed to load robot configuration: {e}")
        return False
    
    task = None
    total_attempts = 5
    total_success = 0
    
    try:
        for attempt_num in range(1, total_attempts + 1):
            # Create or reset task for new attempt
            if task is None:
                # First attempt - create new task
                print("Creating task instance...")
                task = grasp_apple()
                kwargs = create_kwargs(robot_file, embodiment_args)
                task.setup_demo(**kwargs)
                print("✓ Demo environment setup complete")
            else:
                # Subsequent attempts - reset task
                print("🔄 Resetting environment for new attempt...")
                task = reset_task(task, robot_file, embodiment_args)
                if task is None:
                    print("❌ Failed to reset task, exiting...")
                    break
                print("✓ Environment reset complete")
            
            # Run the attempt
            success = run_single_attempt(task, attempt_num)
            if success:
                total_success += 1
            
            # Show statistics
            success_rate = (total_success / attempt_num) * 100
            print(f"\n📊 Statistics: {total_success}/{attempt_num} successful ({success_rate:.1f}%)")
            
            # Wait a bit between attempts
            if attempt_num < total_attempts:
                print("⏳ Waiting 2 seconds before next attempt...")
                time.sleep(2)
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if task is not None:
            try:
                task.close_env()
                print("✓ Environment closed successfully")
            except Exception as e:
                print(f"Warning: Error closing environment: {e}")
    
    print(f"\n🏁 Final Results: {total_success}/{total_attempts} successful attempts ({(total_success/max(total_attempts,1)*100):.1f}%)")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✓ Visualization session completed")
    else:
        print("✗ Visualization session failed")
        sys.exit(1) 