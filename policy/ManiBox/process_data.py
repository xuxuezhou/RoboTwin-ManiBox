import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py
import copy
from glob import glob
from policy.ManiBox.manibox.ManiBox.yolo_process_data import YoloProcessDataByTimeStep
import torch
import subprocess
import tempfile

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        vector = root["/joint_action/vector"][()]
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return left_gripper, left_arm, right_gripper, right_arm, vector, image_dict

def pad_array(array, max_length):
    """
    Pads the input array to the specified max_length with zeros.
    If the array is longer than max_length, it will be truncated.
    """
    if len(array) > max_length:
        return array[:max_length]
    else:
        for _ in range(max_length - len(array)):
            array.append(copy.deepcopy(array[-1]))
        return array

def draw_bboxes_on_image(image, bboxes, colors=None, labels=None):
    """
    Draw bounding boxes on an image.
    
    Args:
        image: numpy array of shape (H, W, 3), BGR format
        bboxes: tensor of shape (total_detections, 4) in xyxyn format (normalized coordinates)
        colors: list of colors for each bounding box
        labels: list of labels for each bounding box
    
    Returns:
        image with bounding boxes drawn
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    
    if len(bboxes.shape) == 1:
        bboxes = bboxes.reshape(1, -1)
    
    h, w = image.shape[:2]
    image_copy = image.copy()
    
    # Default colors (BGR format)
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    # Process bboxes in groups of 4 (xyxyn format)
    num_detections = len(bboxes) // 4 if len(bboxes.shape) == 1 else bboxes.shape[0]
    
    for i in range(num_detections):
        if len(bboxes.shape) == 1:
            bbox = bboxes[i*4:(i+1)*4]
        else:
            bbox = bboxes[i]
        
        # Skip invalid bounding boxes (all zeros)
        if np.allclose(bbox, 0):
            continue
            
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        
        # Choose color
        color = colors[i % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        
        # Draw label if provided
        if labels and i < len(labels):
            label_text = labels[i]
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image_copy, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image_copy

def concatenate_camera_views(images, layout="horizontal"):
    """
    Concatenate multiple camera views into a single image.
    
    Args:
        images: list of images (numpy arrays)
        layout: "horizontal" or "vertical" or "grid"
    
    Returns:
        concatenated image
    """
    if layout == "horizontal":
        return np.concatenate(images, axis=1)
    elif layout == "vertical":
        return np.concatenate(images, axis=0)
    elif layout == "grid":
        # Arrange in a 2x2 grid for 3 cameras (with one empty slot)
        if len(images) == 3:
            # Resize images to same height if needed
            target_height = min(img.shape[0] for img in images)
            target_width = min(img.shape[1] for img in images)
            
            resized_images = []
            for img in images:
                resized = cv2.resize(img, (target_width, target_height))
                resized_images.append(resized)
            
            # Create 2x2 grid
            top_row = np.concatenate([resized_images[0], resized_images[1]], axis=1)
            bottom_row = np.concatenate([resized_images[2], np.zeros_like(resized_images[2])], axis=1)
            return np.concatenate([top_row, bottom_row], axis=0)
    
    return np.concatenate(images, axis=1)  # Default to horizontal

def check_ffmpeg_available():
    """Check if ffmpeg is available in the system."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def create_video_with_ffmpeg(frames, output_path, fps=30, crf=23, temp_dir=None):
    """
    Create video using ffmpeg from a list of frames.
    
    Args:
        frames: list of numpy arrays (frames)
        output_path: path to save the video
        fps: frames per second
        crf: video quality (lower is better quality, 18-28 is good range)
        temp_dir: temporary directory to store frame images
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    try:
        # Save frames as temporary images
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        # Use ffmpeg to create video
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),  # Quality setting
            "-preset", "medium",  # Encoding speed vs compression efficiency
            output_path
        ]
        
        # Run ffmpeg command
        result = subprocess.run(ffmpeg_cmd, 
                              capture_output=True, 
                              text=True, 
                              check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error creating video: {e}")
        return False
    finally:
        # Clean up temporary files
        try:
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    # parser.add_argument(
    #     "object_names",
    #     type=str,
    #     help="Comma-separated list of object names (e.g., obj1,obj2,obj3)",
    # )
    parser.add_argument(
        "--max_episodes",
        type=int,
        help="Number of episodes to process (e.g., 50)",
        default=100000,
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of bounding boxes and generate video output",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualization_output",
        help="Directory to save visualization videos",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30)",
    )
    parser.add_argument(
        "--video_quality",
        type=int,
        default=23,
        help="Video quality (CRF value, lower is better quality, default: 23)",
    )
    args = parser.parse_args()

    load_dir = os.path.join("data", args.task_name)
    if not os.path.exists(load_dir):
        print(f"Directory {load_dir} does not exist.")
        return
    hdf5_files = sorted(glob(os.path.join(load_dir, "**", "*.hdf5"), recursive=True))

    bbox_arrays, state_arrays, joint_action_arrays = [], [], []
    yolo_preprocess_data = YoloProcessDataByTimeStep(objects_names=["bottle"], max_detections_per_object=2)
    max_eps_len, current_ep = 0, 0
    tot_ep = min(len(hdf5_files), args.max_episodes)

    # Create output directory for visualization if needed
    if args.visualize:
        # Check if ffmpeg is available
        if not check_ffmpeg_available():
            print("Error: ffmpeg is not available. Please install ffmpeg to use visualization.")
            print("On Ubuntu/Debian: sudo apt install ffmpeg")
            print("On CentOS/RHEL: sudo yum install ffmpeg")
            print("On macOS: brew install ffmpeg")
            return
        
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Visualization enabled. Videos will be saved to: {args.output_dir}")
        print(f"Video settings: {args.fps} FPS, CRF quality: {args.video_quality}")

    while current_ep < tot_ep:
        print(f"processing episode: {current_ep + 1} / {tot_ep}", end="\r")

        load_path = hdf5_files[current_ep]
        bbox_array, state_array, joint_action_array = [], [], []
        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            vector_all,
            image_dict_all,
        ) = load_hdf5(load_path)

        yolo_preprocess_data.reset_new_episode()
        
        # Initialize video frames list if visualization is enabled
        video_frames = []
        if args.visualize:
            video_path = os.path.join(args.output_dir, f"episode_{current_ep:04d}.mp4")
        
        for j in range(0, left_gripper_all.shape[0]):
            joint_state = vector_all[j]

            if j != left_gripper_all.shape[0] - 1:
                images = []
                for camera in ["head_camera", "left_camera", "right_camera"]:
                    images.append(cv2.imdecode(np.frombuffer(image_dict_all[camera][j], np.uint8), cv2.IMREAD_COLOR))
                
                # Get bounding box predictions
                bbox_result = yolo_preprocess_data.process(*images)
                bbox_array.append(bbox_result.squeeze(0)) 
                state_array.append(joint_state)
                
                # Visualization
                if args.visualize:
                    # Draw bounding boxes on each camera view
                    images_with_bbox = []
                    camera_names = ["head_camera", "left_camera", "right_camera"]
                    
                    # Extract bboxes for each camera (assuming 3 cameras, 2 detections each, 4 coords each = 24 total)
                    # bbox_result is tensor of shape (1, 24) -> (24,)
                    if isinstance(bbox_result, torch.Tensor):
                        bbox_data = bbox_result.squeeze().cpu().numpy()
                    else:
                        bbox_data = np.array(bbox_result).flatten()
                    
                    # Split bbox data for each camera (8 values per camera: 2 detections * 4 coords)
                    detections_per_camera = 2
                    coords_per_detection = 4
                    values_per_camera = detections_per_camera * coords_per_detection
                    
                    for cam_idx, (image, cam_name) in enumerate(zip(images, camera_names)):
                        start_idx = cam_idx * values_per_camera
                        end_idx = start_idx + values_per_camera
                        cam_bboxes = bbox_data[start_idx:end_idx].reshape(detections_per_camera, coords_per_detection)
                        
                        # Draw bounding boxes on this camera's image
                        image_with_bbox = draw_bboxes_on_image(
                            image, 
                            cam_bboxes, 
                            labels=[f"bottle_{i+1}" for i in range(detections_per_camera)]
                        )
                        
                        # Add camera label
                        cv2.putText(image_with_bbox, cam_name, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        images_with_bbox.append(image_with_bbox)
                    
                    # Concatenate camera views
                    concatenated_frame = concatenate_camera_views(images_with_bbox, layout="horizontal")
                    
                    # Add episode and frame information
                    info_text = f"Episode: {current_ep+1}, Frame: {j+1}/{left_gripper_all.shape[0]-1}"
                    cv2.putText(concatenated_frame, info_text, (10, concatenated_frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Store frame for later video creation
                    video_frames.append(concatenated_frame)
                    
            if j != 0:
                joint_action_array.append(joint_state)
        
        # Create video using ffmpeg if visualization is enabled
        if args.visualize and video_frames:
            print(f"\nCreating video for episode {current_ep+1}...")
            success = create_video_with_ffmpeg(video_frames, video_path, 
                                             fps=args.fps, crf=args.video_quality)
            if success:
                print(f"Saved visualization video: {video_path}")
            else:
                print(f"Failed to create video: {video_path}")
            # Clear frames to free memory
            video_frames.clear()

        current_ep += 1
        max_eps_len = max(max_eps_len, left_gripper_all.shape[0] - 1)
        bbox_arrays.append(bbox_array)
        state_arrays.append(state_array)
        joint_action_arrays.append(joint_action_array)
    
    # pad the arrays
    for i in range(len(bbox_arrays)):
        bbox_arrays[i] = pad_array(bbox_arrays[i], max_eps_len)
        state_arrays[i] = pad_array(state_arrays[i], max_eps_len)
        joint_action_arrays[i] = pad_array(joint_action_arrays[i], max_eps_len)

    bbox_arrays = np.array(bbox_arrays)
    state_arrays = np.array(state_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    print("Shape of processed data:")
    print(bbox_arrays.shape, state_arrays.shape, joint_action_arrays.shape)

    torch.save({
        "image_data": torch.from_numpy(bbox_arrays),
        "qpos_data": torch.from_numpy(state_arrays),
        "action_data": torch.from_numpy(joint_action_arrays)
    }, os.path.join(load_dir, "manibox_data.pkl"))
    
    if args.visualize:
        print(f"Visualization complete. {current_ep} videos saved to {args.output_dir}")

if __name__ == "__main__":
    main()
