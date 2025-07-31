#!/usr/bin/env python3
"""
打印原始数据脚本
打印action、state和bbox的实际数据值，全部记录在同一个log文件里面
"""

import os
import sys
import torch
import numpy as np
import logging
from datetime import datetime

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
manibox_dir = os.path.join(current_dir, "policy/ManiBox/manibox/ManiBox")
if manibox_dir not in sys.path:
    sys.path.insert(0, manibox_dir)


def setup_logging(log_file):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def print_episode_data(data, episode_id, logger):
    """打印单个episode的详细数据"""
    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 Episode {episode_id} 原始数据")
    logger.info(f"{'='*80}")
    
    # 获取episode数据
    if 'image_data' in data:
        image_data = data['image_data'][episode_id]  # [170, 24]
        logger.info(f"📷 BBox数据 (Episode {episode_id}):")
        logger.info(f"   形状: {image_data.shape}")
        logger.info(f"   数据类型: {image_data.dtype}")
        
        # 解析bbox数据 (3个相机，每个相机2个检测，每个检测4个坐标)
        cameras = ['head_camera', 'left_camera', 'right_camera']
        
        # 打印前10个时间步的详细数据
        for t in range(min(10, image_data.shape[0])):
            logger.info(f"\n   时间步 {t}:")
            bbox_data = image_data[t]  # [24]
            
            for cam_idx, cam_name in enumerate(cameras):
                for det_idx in range(2):
                    start_idx = cam_idx * 8 + det_idx * 4
                    bbox = bbox_data[start_idx:start_idx+4].tolist()
                    logger.info(f"     {cam_name}, 检测{det_idx}: [{bbox[0]:.6f}, {bbox[1]:.6f}, {bbox[2]:.6f}, {bbox[3]:.6f}]")
        
        # 打印所有时间步的bbox数据（简化格式）
        logger.info(f"\n   所有时间步的BBox数据:")
        for t in range(image_data.shape[0]):
            bbox_data = image_data[t]
            bbox_str = ", ".join([f"{x:.6f}" for x in bbox_data.tolist()])
            logger.info(f"     T{t:03d}: [{bbox_str}]")
    
    if 'qpos_data' in data:
        qpos_data = data['qpos_data'][episode_id]  # [170, 14]
        logger.info(f"\n🤖 关节位置数据 (Episode {episode_id}):")
        logger.info(f"   形状: {qpos_data.shape}")
        logger.info(f"   数据类型: {qpos_data.dtype}")
        
        # 打印前10个时间步的详细数据
        for t in range(min(10, qpos_data.shape[0])):
            joint_data = qpos_data[t].tolist()
            joint_str = ", ".join([f"{x:.6f}" for x in joint_data])
            logger.info(f"    T{t:03d}: [{joint_str}]")
        
        # 打印所有时间步的关节数据
        logger.info(f"\n   所有时间步的关节位置数据:")
        for t in range(qpos_data.shape[0]):
            joint_data = qpos_data[t].tolist()
            joint_str = ", ".join([f"{x:.6f}" for x in joint_data])
            logger.info(f"     T{t:03d}: [{joint_str}]")
    
    if 'action_data' in data:
        action_data = data['action_data'][episode_id]  # [170, 14]
        logger.info(f"\n🎯 动作数据 (Episode {episode_id}):")
        logger.info(f"   形状: {action_data.shape}")
        logger.info(f"   数据类型: {action_data.dtype}")
        
        # 打印前10个时间步的详细数据
        for t in range(min(10, action_data.shape[0])):
            action_values = action_data[t].tolist()
            action_str = ", ".join([f"{x:.6f}" for x in action_values])
            logger.info(f"    T{t:03d}: [{action_str}]")
        
        # 打印所有时间步的动作数据
        logger.info(f"\n   所有时间步的动作数据:")
        for t in range(action_data.shape[0]):
            action_values = action_data[t].tolist()
            action_str = ", ".join([f"{x:.6f}" for x in action_values])
            logger.info(f"     T{t:03d}: [{action_str}]")


def print_all_data(data_path, logger):
    """打印所有数据"""
    logger.info("🚀 开始打印原始数据")
    logger.info(f"数据文件: {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        return
    
    try:
        # 加载数据
        data = torch.load(data_path, map_location='cpu')
        logger.info("✅ 数据加载成功")
        
        # 基本信息
        logger.info(f"\n📊 数据基本信息:")
        logger.info(f"   数据键: {list(data.keys())}")
        
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"   {key}: 形状={value.shape}, 类型={value.dtype}")
        
        # 打印每个episode的数据
        num_episodes = data['image_data'].shape[0]
        logger.info(f"\n📈 总共 {num_episodes} 个episode")
        
        # 打印所有episode的数据
        for episode_id in range(num_episodes):
            print_episode_data(data, episode_id, logger)
        
        logger.info(f"\n✅ 所有数据打印完成!")
        
    except Exception as e:
        logger.error(f"❌ 数据加载过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """主函数"""
    # 数据文件路径
    data_path = "/home/xuxuezhou/code/RoboTwin/data/move_can_pot/integration.pkl"
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"raw_data_{timestamp}.log"
    logger = setup_logging(log_file)
    
    print(f"🚀 开始打印原始数据，日志文件: {log_file}")
    
    # 打印所有数据
    print_all_data(data_path, logger)
    
    print(f"✅ 数据打印完成，日志文件: {log_file}")


if __name__ == "__main__":
    main() 