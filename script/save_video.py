import subprocess
import pickle, os, pdb

from pathlib import Path

folder_path = f'data/shoe_place_mt1_rt1_pkl/episode0/'

save_dir = Path('data_video/')
save_dir.mkdir(parents=True, exist_ok=True)
ffmpeg = subprocess.Popen([
    'ffmpeg', '-y',
    '-f', 'rawvideo',
    '-pixel_format', 'rgb24',
    '-video_size', '320x240',
    '-framerate', '10',
    '-i', '-',
    '-pix_fmt', 'yuv420p',
    '-vcodec', 'libx264',
    '-crf', '23',
    f'{save_dir}' + '/tmp.mp4'
], stdin=subprocess.PIPE)


def count_files_in_directory(directory_path):
    try:
        items = os.listdir(directory_path)
        file_count = sum(1 for item in items if os.path.isfile(os.path.join(directory_path, item)))
        return file_count
    except FileNotFoundError:
        print(f"目录 {directory_path} 不存在")
        return None
    except PermissionError:
        print(f"没有权限访问目录 {directory_path}")
        return None

def read_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def get_obs_shoufa():
    obs_list, action_list = [], []
    obs_num = count_files_in_directory(folder_path)
    for i in range(obs_num):
        file_path = folder_path + f'{i}.pkl'
        obs = read_pkl(file_path)
        obs_list.append(obs)
        action_list.append(obs['joint_action'])
    return obs_list, action_list

gt_obs, gt_actions = get_obs_shoufa()
for i in range(len(gt_obs)):
    rgb = gt_obs[i]['observation']['head_camera']['rgb']
    print(rgb.shape)
    ffmpeg.stdin.write(rgb.tobytes())


ffmpeg.stdin.close()
ffmpeg.wait()
