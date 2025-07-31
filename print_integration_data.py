#!/usr/bin/env python3
"""
读取和打印integration.pkl数据的脚本
参考process_data.py的数据格式
"""

import os
import pickle
import torch
import numpy as np
from pathlib import Path

def load_and_print_integration_data(data_path):
    """
    加载并打印integration.pkl数据
    
    Args:
        data_path: integration.pkl文件的路径
    """
    print("🔍 正在加载数据...")
    print(f"📁 数据路径: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 错误: 文件不存在 - {data_path}")
        return
    
    try:
        # 加载数据
        with open(data_path, 'rb') as f:
            data = torch.load(f, map_location='cpu')
        
        print("✅ 数据加载成功!")
        print("\n" + "="*60)
        
        # 打印数据基本信息
        print("📊 数据基本信息:")
        print(f"   数据类型: {type(data)}")
        print(f"   数据键: {list(data.keys())}")
        
        # 分析每个数据组件
        for key, value in data.items():
            print(f"\n🔍 分析 {key}:")
            
            if isinstance(value, torch.Tensor):
                print(f"   类型: torch.Tensor")
                print(f"   形状: {value.shape}")
                print(f"   数据类型: {value.dtype}")
                print(f"   设备: {value.device}")
                print(f"   数值范围: [{value.min():.6f}, {value.max():.6f}]")
                print(f"   均值: {value.mean():.6f}")
                print(f"   标准差: {value.std():.6f}")
                
                # 检查是否有NaN或无穷大值
                if torch.isnan(value).any():
                    print(f"   ⚠️  包含NaN值: {torch.isnan(value).sum()} 个")
                if torch.isinf(value).any():
                    print(f"   ⚠️  包含无穷大值: {torch.isinf(value).sum()} 个")
                
                # 显示前几个样本的数据
                if len(value.shape) >= 2:
                    print(f"   前3个样本的前5个值:")
                    for i in range(min(3, value.shape[0])):
                        sample_data = value[i]
                        if len(sample_data.shape) >= 1:
                            print(f"     样本{i+1}: {sample_data.flatten()[:5].tolist()}")
                
            elif isinstance(value, np.ndarray):
                print(f"   类型: numpy.ndarray")
                print(f"   形状: {value.shape}")
                print(f"   数据类型: {value.dtype}")
                print(f"   数值范围: [{value.min():.6f}, {value.max():.6f}]")
                print(f"   均值: {value.mean():.6f}")
                print(f"   标准差: {value.std():.6f}")
                
            else:
                print(f"   类型: {type(value)}")
                print(f"   内容: {value}")
        
        # 详细分析数据内容
        print("\n" + "="*60)
        print("📈 详细数据分析:")
        
        # 分析episode数量
        if 'image_data' in data:
            num_episodes = data['image_data'].shape[0]
            print(f"   总episode数量: {num_episodes}")
        
        # 分析时间步长
        if 'qpos_data' in data:
            max_timesteps = data['qpos_data'].shape[1]
            print(f"   最大时间步长: {max_timesteps}")
        
        # 分析动作维度
        if 'action_data' in data:
            action_dim = data['action_data'].shape[-1]
            print(f"   动作维度: {action_dim}")
        
        # 分析图像数据维度
        if 'image_data' in data:
            image_dim = data['image_data'].shape[-1]
            print(f"   图像数据维度: {image_dim}")
            print(f"   每个episode的图像数据形状: {data['image_data'].shape[1:]}")
        
        # 分析状态数据维度
        if 'qpos_data' in data:
            state_dim = data['qpos_data'].shape[-1]
            print(f"   状态数据维度: {state_dim}")
        
        # 检查数据完整性
        print("\n🔍 数据完整性检查:")
        if 'image_data' in data and 'qpos_data' in data and 'action_data' in data:
            img_episodes = data['image_data'].shape[0]
            qpos_episodes = data['qpos_data'].shape[0]
            action_episodes = data['action_data'].shape[0]
            
            if img_episodes == qpos_episodes == action_episodes:
                print(f"   ✅ 所有数据episode数量一致: {img_episodes}")
            else:
                print(f"   ❌ episode数量不一致:")
                print(f"      图像数据: {img_episodes}")
                print(f"      状态数据: {qpos_episodes}")
                print(f"      动作数据: {action_episodes}")
        
        # 显示一些样本数据
        print("\n" + "="*60)
        print("📋 样本数据展示:")
        
        if 'image_data' in data:
            print(f"\n🎯 图像数据样本 (前3个episode, 前2个时间步):")
            for ep in range(min(3, data['image_data'].shape[0])):
                print(f"   Episode {ep+1}:")
                for t in range(min(2, data['image_data'].shape[1])):
                    sample = data['image_data'][ep, t]
                    print(f"     时间步 {t+1}: {sample.flatten()[:10].tolist()}...")
        
        if 'qpos_data' in data:
            print(f"\n🤖 状态数据样本 (前3个episode, 前2个时间步):")
            for ep in range(min(3, data['qpos_data'].shape[0])):
                print(f"   Episode {ep+1}:")
                for t in range(min(2, data['qpos_data'].shape[1])):
                    sample = data['qpos_data'][ep, t]
                    print(f"     时间步 {t+1}: {sample.flatten()[:10].tolist()}...")
        
        if 'action_data' in data:
            print(f"\n🎮 动作数据样本 (前3个episode, 前2个时间步):")
            for ep in range(min(3, data['action_data'].shape[0])):
                print(f"   Episode {ep+1}:")
                for t in range(min(2, data['action_data'].shape[1])):
                    sample = data['action_data'][ep, t]
                    print(f"     时间步 {t+1}: {sample.flatten()[:10].tolist()}...")
        
        print("\n" + "="*60)
        print("✅ 数据分析完成!")
        
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🎯 Integration.pkl 数据分析工具")
    print("="*60)
    
    # 数据路径
    data_path = "/home/xuxuezhou/code/RoboTwin/data/move_can_pot/integration.pkl"
    
    # 检查文件是否存在
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        print("请检查路径是否正确")
        return
    
    # 加载并打印数据
    load_and_print_integration_data(data_path)

if __name__ == "__main__":
    main() 