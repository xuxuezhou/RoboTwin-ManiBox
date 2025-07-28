# ManiBox Data Processing

ManiBox是一个用于处理ACT格式数据并结合YOLO目标检测的数据处理工具。它将原始的HDF5图像数据转换为边界框(bbox)特征，用于机器人学习任务。

## 文件说明

- `manibox_data_processor.py`: 主要的数据处理脚本（GPU内存优化版）
- `run_manibox.sh`: 便捷的运行脚本
- `GPU_MEMORY_GUIDE.md`: GPU显存问题解决指南
- `README_Manibox.md`: 本说明文档

## 功能特点

1. **ACT格式兼容**: 读取ACT框架的HDF5数据格式
2. **YOLO目标检测**: 使用YOLO模型检测指定目标物体
3. **边界框输出**: 输出归一化的边界框坐标而非原始图像
4. **多相机支持**: 支持头部相机、左手腕相机、右手腕相机
5. **多目标检测**: 可同时检测多个目标物体
6. **卡尔曼滤波**: 可选的卡尔曼滤波平滑边界框轨迹

## 使用方法

### 方法1: 使用Shell脚本（推荐）

首先使脚本可执行：
```bash
chmod +x run_manibox.sh
```

然后运行：
```bash
./run_manibox.sh <task_name> <task_config> <num_episodes> [batch_size] [use_cpu]
```

示例：
```bash
# 基本用法（默认GPU批次4）
./run_manibox.sh grasp_apple default 50

# 减小批次大小（解决GPU显存不足）
./run_manibox.sh grasp_apple default 50 2

# 使用CPU处理（最安全但较慢）
./run_manibox.sh grasp_apple default 50 4 cpu
```

### 方法2: 直接运行Python脚本

```bash
# 基本用法
python manibox_data_processor.py grasp_apple default 50

# GPU显存优化
python manibox_data_processor.py grasp_apple default 50 --batch_size 2

# 强制CPU处理
python manibox_data_processor.py grasp_apple default 50 --cpu

# 检测多个对象
python manibox_data_processor.py grasp_apple default 50 --objects apple,table
```

### GPU显存不足？

如果遇到CUDA显存错误，请参考 `GPU_MEMORY_GUIDE.md` 或尝试：
1. 减小批次大小：`--batch_size 2` 或 `--batch_size 1`
2. 使用CPU模式：`--cpu`

## 参数说明

- `task_name`: 任务名称（如：grasp_apple）
- `task_config`: 任务配置（如：default）
- `num_episodes`: 要处理的episode数量
- `batch_size`: YOLO处理批次大小（默认：4，显存不足时减小）
- `use_cpu`: 强制使用CPU处理（填写"cpu"）
- `--objects`: 要检测的目标物体，用逗号分隔（默认：apple）
- `--cpu`: 强制CPU处理标志
- `--batch_size`: 指定批次大小

## 输入数据格式

输入数据应位于：`../../data/{task_name}/{task_config}/data/`

期望的HDF5文件结构：
```
episode{i}.hdf5
├── /joint_action/
│   ├── left_gripper
│   ├── left_arm
│   ├── right_gripper
│   └── right_arm
└── /observation/
    ├── head_camera/rgb
    ├── left_camera/rgb
    └── right_camera/rgb
```

## 输出数据格式

输出保存在：`processed_data/manibox-{task_name}/{task_config}-{num_episodes}/integration.pkl`

输出的数据字典包含：
- `image_data`: 边界框数据，形状为 (num_episodes, episode_len, bbox_dim)
- `qpos_data`: 关节位置数据，形状为 (num_episodes, episode_len, 14)
- `action_data`: 动作数据，形状为 (num_episodes, episode_len, 14)
- `reward`: 奖励数据（占位符）

其中 `bbox_dim = num_objects * num_cameras * 4`，每个边界框包含4个归一化坐标[x1, y1, x2, y2]。

## 相机映射

- `head_camera` → `cam_high`
- `left_camera` → `cam_left_wrist`
- `right_camera` → `cam_right_wrist`

## 依赖项

- torch
- h5py
- numpy
- opencv-python
- ultralytics (YOLO)
- tqdm

## 注意事项

1. 确保有足够的存储空间处理大量数据
2. YOLO模型会在首次运行时自动下载
3. 处理时间取决于数据量和计算资源
4. 如果episode长度不一致，会自动进行填充或截断处理

## 故障排除

1. **路径错误**: 确保数据路径正确，特别是相对路径设置
2. **内存不足**: 减少batch_size或处理更少的episodes
3. **YOLO模型下载失败**: 检查网络连接，可能需要手动下载模型
4. **依赖项缺失**: 使用pip安装所需的依赖项

## 配置文件

处理完成后，会在当前目录生成`MANIBOX_TASK_CONFIGS.json`配置文件，包含处理后数据的详细信息，可用于后续的模型训练。 