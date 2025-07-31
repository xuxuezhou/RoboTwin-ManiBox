#!/usr/bin/env python3
"""
快速查看integration.pkl数据总结
"""

import os
import torch
import numpy as np

def quick_data_summary(data_path):
    """
    快速数据总结
    
    Args:
        data_path: integration.pkl文件的路径
    """
    print("📊 数据快速总结")
    print("="*50)
    
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return
    
    try:
        # 加载数据
        with open(data_path, 'rb') as f:
            data = torch.load(f, map_location='cpu')
        
        print(f"✅ 数据加载成功!")
        print(f"📁 文件路径: {data_path}")
        print(f"📦 数据键: {list(data.keys())}")
        
        # 基本信息
        print(f"\n📈 数据形状:")
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape} ({value.dtype})")
            else:
                print(f"   {key}: {type(value)}")
        
        # 数据统计
        print(f"\n📊 数据统计:")
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}:")
                print(f"     范围: [{value.min():.4f}, {value.max():.4f}]")
                print(f"     均值: {value.mean():.4f}")
                print(f"     标准差: {value.std():.4f}")
                if torch.isnan(value).any():
                    print(f"     ⚠️  NaN值: {torch.isnan(value).sum()}")
                if torch.isinf(value).any():
                    print(f"     ⚠️  无穷大值: {torch.isinf(value).sum()}")
        
        # 图像数据详细分析
        if 'image_data' in data:
            image_data = data['image_data']
            print(f"\n🎯 图像数据详细分析:")
            print(f"   Episode数量: {image_data.shape[0]}")
            print(f"   时间步数量: {image_data.shape[1]}")
            print(f"   特征维度: {image_data.shape[2]}")
            
            # 分析每个相机的检测情况
            cameras = ["head_camera", "left_camera", "right_camera"]
            for cam_idx, cam_name in enumerate(cameras):
                start_idx = cam_idx * 8
                end_idx = start_idx + 8
                cam_data = image_data[:, :, start_idx:end_idx]
                
                # 计算有效检测率
                valid_detections = (cam_data != 0).any(dim=2).float().mean()
                print(f"   {cam_name} 有效检测率: {valid_detections:.2%}")
        
        # 状态和动作数据
        if 'qpos_data' in data and 'action_data' in data:
            print(f"\n🤖 状态和动作数据:")
            print(f"   状态维度: {data['qpos_data'].shape[-1]}")
            print(f"   动作维度: {data['action_data'].shape[-1]}")
            
            # 检查状态和动作是否相同
            if torch.allclose(data['qpos_data'], data['action_data']):
                print(f"   ⚠️  状态和动作数据完全相同")
            else:
                print(f"   ✅ 状态和动作数据不同")
        
        print(f"\n" + "="*50)
        print("✅ 数据总结完成!")
        
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")

def main():
    """主函数"""
    data_path = "/home/xuxuezhou/code/RoboTwin/data/move_can_pot/integration.pkl"
    quick_data_summary(data_path)

if __name__ == "__main__":
    main() 