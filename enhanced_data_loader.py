#!/usr/bin/env python3
"""
增强的数据读取脚本
添加详细的数据打印功能，将所有数据信息保存到log文件
"""

import os
import sys
import torch
import numpy as np
import json
import logging
from datetime import datetime
from collections import defaultdict

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
manibox_dir = os.path.join(current_dir, "policy/ManiBox/manibox/ManiBox")
if manibox_dir not in sys.path:
    sys.path.insert(0, manibox_dir)

from dataloader.BBoxHistoryEpisodicDataset import BBoxHistoryEpisodicDataset
from dataloader.data_load import get_norm_stats, load_data


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


def analyze_data_structure(data, logger):
    """分析数据结构"""
    logger.info("="*80)
    logger.info("📊 数据结构分析")
    logger.info("="*80)
    
    # 基本信息
    logger.info(f"数据键: {list(data.keys())}")
    logger.info(f"数据类型: {type(data)}")
    
    # 分析每个键的数据
    for key, value in data.items():
        logger.info(f"\n🔍 分析键: {key}")
        if isinstance(value, torch.Tensor):
            logger.info(f"   形状: {value.shape}")
            logger.info(f"   数据类型: {value.dtype}")
            logger.info(f"   设备: {value.device}")
            logger.info(f"   数值范围: [{value.min():.6f}, {value.max():.6f}]")
            logger.info(f"   均值: {value.mean():.6f}")
            logger.info(f"   标准差: {value.std():.6f}")
            logger.info(f"   非零元素数量: {(value != 0).sum().item()}")
            logger.info(f"   总元素数量: {value.numel()}")
            logger.info(f"   非零比例: {(value != 0).float().mean():.2%}")
            
            # 如果是多维数据，分析每个维度
            if len(value.shape) > 1:
                for dim in range(len(value.shape)):
                    dim_data = value.mean(dim=tuple(i for i in range(len(value.shape)) if i != dim))
                    logger.info(f"   维度{dim}统计: 范围=[{dim_data.min():.6f}, {dim_data.max():.6f}], 均值={dim_data.mean():.6f}")
        else:
            logger.info(f"   类型: {type(value)}")
            logger.info(f"   值: {value}")


def analyze_episode_data(data, episode_id, logger):
    """分析单个episode的数据"""
    logger.info(f"\n🎯 Episode {episode_id} 详细分析")
    logger.info("-"*60)
    
    # 获取episode数据
    if 'image_data' in data:
        image_data = data['image_data'][episode_id]
        logger.info(f"图像数据形状: {image_data.shape}")
        
        # 分析bbox数据
        if len(image_data.shape) >= 2:
            bbox_dim = image_data.shape[-1]
            logger.info(f"BBox维度: {bbox_dim}")
            
            # 分析每个时间步的bbox
            for t in range(min(5, image_data.shape[0])):  # 只分析前5个时间步
                bbox_data = image_data[t]
                logger.info(f"  时间步 {t}: bbox数据范围=[{bbox_data.min():.6f}, {bbox_data.max():.6f}]")
                
                # 解析bbox (假设是3个相机，每个相机2个检测，每个检测4个坐标)
                if bbox_dim == 24:  # 3*2*4
                    for cam_idx, cam_name in enumerate(['head', 'left_wrist', 'right_wrist']):
                        for det_idx in range(2):
                            start_idx = cam_idx * 8 + det_idx * 4
                            bbox = bbox_data[start_idx:start_idx+4].tolist()
                            logger.info(f"    {cam_name}_cam, 检测{det_idx}: {bbox}")
    
    if 'qpos_data' in data:
        qpos_data = data['qpos_data'][episode_id]
        logger.info(f"关节位置数据形状: {qpos_data.shape}")
        logger.info(f"关节位置范围: [{qpos_data.min():.6f}, {qpos_data.max():.6f}]")
        
        # 分析每个关节
        for joint_idx in range(min(5, qpos_data.shape[1])):  # 只分析前5个关节
            joint_data = qpos_data[:, joint_idx]
            logger.info(f"  关节 {joint_idx}: 范围=[{joint_data.min():.6f}, {joint_data.max():.6f}], 均值={joint_data.mean():.6f}")
    
    if 'action_data' in data:
        action_data = data['action_data'][episode_id]
        logger.info(f"动作数据形状: {action_data.shape}")
        logger.info(f"动作范围: [{action_data.min():.6f}, {action_data.max():.6f}]")
        
        # 分析每个动作维度
        for action_idx in range(min(5, action_data.shape[1])):  # 只分析前5个动作维度
            action_dim_data = action_data[:, action_idx]
            logger.info(f"  动作维度 {action_idx}: 范围=[{action_dim_data.min():.6f}, {action_dim_data.max():.6f}], 均值={action_dim_data.mean():.6f}")


def analyze_dataset_statistics(data, logger):
    """分析数据集统计信息"""
    logger.info("\n📈 数据集统计信息")
    logger.info("="*60)
    
    # 计算基本统计信息
    stats = {}
    
    if 'image_data' in data:
        image_data = data['image_data']
        stats['image_data'] = {
            'shape': list(image_data.shape),
            'min': image_data.min().item(),
            'max': image_data.max().item(),
            'mean': image_data.mean().item(),
            'std': image_data.std().item(),
            'nonzero_ratio': (image_data != 0).float().mean().item()
        }
        logger.info(f"图像数据统计: {stats['image_data']}")
    
    if 'qpos_data' in data:
        qpos_data = data['qpos_data']
        stats['qpos_data'] = {
            'shape': list(qpos_data.shape),
            'min': qpos_data.min().item(),
            'max': qpos_data.max().item(),
            'mean': qpos_data.mean().item(),
            'std': qpos_data.std().item()
        }
        logger.info(f"关节位置统计: {stats['qpos_data']}")
    
    if 'action_data' in data:
        action_data = data['action_data']
        stats['action_data'] = {
            'shape': list(action_data.shape),
            'min': action_data.min().item(),
            'max': action_data.max().item(),
            'mean': action_data.mean().item(),
            'std': action_data.std().item()
        }
        logger.info(f"动作数据统计: {stats['action_data']}")
    
    return stats


def analyze_dataloader(dataloader, logger):
    """分析数据加载器"""
    logger.info("\n🔄 数据加载器分析")
    logger.info("="*60)
    
    logger.info(f"数据加载器长度: {len(dataloader)}")
    
    # 获取第一个batch进行分析
    try:
        first_batch = next(iter(dataloader))
        logger.info(f"第一个batch类型: {type(first_batch)}")
        
        if isinstance(first_batch, (list, tuple)):
            logger.info(f"第一个batch包含 {len(first_batch)} 个元素")
            for i, item in enumerate(first_batch):
                if isinstance(item, torch.Tensor):
                    logger.info(f"  元素 {i}: 形状={item.shape}, 类型={item.dtype}, 范围=[{item.min():.6f}, {item.max():.6f}]")
                else:
                    logger.info(f"  元素 {i}: 类型={type(item)}")
        elif isinstance(first_batch, torch.Tensor):
            logger.info(f"第一个batch: 形状={first_batch.shape}, 类型={first_batch.dtype}")
            logger.info(f"  范围: [{first_batch.min():.6f}, {first_batch.max():.6f}]")
            logger.info(f"  均值: {first_batch.mean():.6f}")
            logger.info(f"  标准差: {first_batch.std():.6f}")
    except Exception as e:
        logger.error(f"分析数据加载器时出错: {e}")


def enhanced_load_data(dataset_dir, num_episodes, arm_delay_time, max_pos_lookahead, 
                      use_dataset_action, use_depth_image, use_robot_base, camera_names, 
                      batch_size_train, batch_size_val, episode_begin=0, episode_end=-1,
                      context_len=1, prefetch_factor=2, dataset_type=BBoxHistoryEpisodicDataset):
    """增强的数据加载函数，包含详细的数据分析"""
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"data_analysis_{timestamp}.log"
    logger = setup_logging(log_file)
    
    logger.info("🚀 开始增强数据加载分析")
    logger.info(f"数据集路径: {dataset_dir}")
    logger.info(f"Episode数量: {num_episodes}")
    logger.info(f"相机名称: {camera_names}")
    
    # 加载原始数据
    data_path = os.path.join(dataset_dir, "integration.pkl")
    logger.info(f"加载数据文件: {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        return None, None, None, None
    
    try:
        data = torch.load(data_path, map_location='cpu')
        logger.info("✅ 数据加载成功")
        
        # 分析数据结构
        analyze_data_structure(data, logger)
        
        # 分析数据集统计信息
        stats = analyze_dataset_statistics(data, logger)
        
        # 分析前几个episode的详细数据
        for episode_id in range(min(3, num_episodes)):
            analyze_episode_data(data, episode_id, logger)
        
        # 计算标准化统计信息
        logger.info("\n📊 计算标准化统计信息")
        logger.info("="*60)
        
        norm_stats = get_norm_stats(dataset_dir, num_episodes, episode_begin, episode_end)
        logger.info(f"标准化统计信息: {norm_stats}")
        
        # 创建数据加载器
        logger.info("\n🔄 创建数据加载器")
        logger.info("="*60)
        
        train_dataloader, val_dataloader, norm_stats, is_sim = load_data(
            dataset_dir, num_episodes, arm_delay_time, max_pos_lookahead,
            use_dataset_action, use_depth_image, use_robot_base, camera_names,
            batch_size_train, batch_size_val, episode_begin, episode_end,
            context_len, prefetch_factor, dataset_type
        )
        
        # 分析数据加载器
        logger.info("\n📊 分析训练数据加载器")
        analyze_dataloader(train_dataloader, logger)
        
        logger.info("\n📊 分析验证数据加载器")
        analyze_dataloader(val_dataloader, logger)
        
        # 保存统计信息到JSON文件
        stats_file = f"data_stats_{timestamp}.json"
        
        # 转换numpy数组为列表以便JSON序列化
        norm_stats_serializable = {}
        for key, value in norm_stats.items():
            if isinstance(value, np.ndarray):
                norm_stats_serializable[key] = value.tolist()
            else:
                norm_stats_serializable[key] = value
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset_info': {
                    'dataset_dir': dataset_dir,
                    'num_episodes': num_episodes,
                    'camera_names': camera_names,
                    'batch_size_train': batch_size_train,
                    'batch_size_val': batch_size_val
                },
                'data_statistics': stats,
                'normalization_stats': norm_stats_serializable,
                'is_simulation': is_sim
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 统计信息已保存到: {stats_file}")
        logger.info(f"✅ 详细日志已保存到: {log_file}")
        
        return train_dataloader, val_dataloader, norm_stats, is_sim
        
    except Exception as e:
        logger.error(f"❌ 数据加载过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None


def main():
    """主函数 - 测试增强数据加载"""
    # 测试参数
    dataset_dir = "/home/xuxuezhou/code/RoboTwin/data/move_can_pot"
    num_episodes = 50
    camera_names = ['head_camera', 'left_camera', 'right_camera']
    batch_size_train = 32
    batch_size_val = 32
    
    print("🚀 开始增强数据加载测试")
    
    # 调用增强的数据加载函数
    train_dataloader, val_dataloader, norm_stats, is_sim = enhanced_load_data(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        arm_delay_time=0,
        max_pos_lookahead=0,
        use_dataset_action=True,
        use_depth_image=False,
        use_robot_base=False,
        camera_names=camera_names,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        episode_begin=0,
        episode_end=-1,
        context_len=1,
        prefetch_factor=2,
        dataset_type=BBoxHistoryEpisodicDataset
    )
    
    if train_dataloader is not None:
        print("✅ 数据加载成功!")
        print(f"训练数据加载器长度: {len(train_dataloader)}")
        print(f"验证数据加载器长度: {len(val_dataloader)}")
    else:
        print("❌ 数据加载失败!")


if __name__ == "__main__":
    main() 