#!/usr/bin/env python3
"""
专门打印动作和状态数据的脚本 - 增强版
"""

import os
import torch
import numpy as np

def print_action_state_data(data_path):
    """
    打印动作和状态数据
    
    Args:
        data_path: integration.pkl文件的路径
    """
    print("🤖 动作和状态数据打印 - 增强版")
    print("="*60)
    
    if not os.path.exists(data_path):
        print(f"❌ 文件不存在: {data_path}")
        return
    
    try:
        # 加载数据
        with open(data_path, 'rb') as f:
            data = torch.load(f, map_location='cpu')
        
        print("✅ 数据加载成功!")
        
        # 获取动作和状态数据
        qpos_data = data['qpos_data']  # 状态数据
        action_data = data['action_data']  # 动作数据
        
        print(f"\n📊 数据基本信息:")
        print(f"   状态数据形状: {qpos_data.shape}")
        print(f"   动作数据形状: {action_data.shape}")
        print(f"   数据类型: {qpos_data.dtype}")
        print(f"   Episode数量: {qpos_data.shape[0]}")
        print(f"   时间步数量: {qpos_data.shape[1]}")
        print(f"   状态/动作维度: {qpos_data.shape[2]}")
        
        # 数据统计
        print(f"\n📈 数据统计:")
        print(f"   状态数据范围: [{qpos_data.min():.6f}, {qpos_data.max():.6f}]")
        print(f"   状态数据均值: {qpos_data.mean():.6f}")
        print(f"   状态数据标准差: {qpos_data.std():.6f}")
        print(f"   动作数据范围: [{action_data.min():.6f}, {action_data.max():.6f}]")
        print(f"   动作数据均值: {action_data.mean():.6f}")
        print(f"   动作数据标准差: {action_data.std():.6f}")
        
        # 检查状态和动作是否相同
        if torch.allclose(qpos_data, action_data):
            print(f"   ⚠️  状态和动作数据完全相同")
        else:
            print(f"   ✅ 状态和动作数据不同")
            diff = torch.abs(qpos_data - action_data)
            print(f"   最大差异: {diff.max():.6f}")
            print(f"   平均差异: {diff.mean():.6f}")
        
        # 打印更多episode的详细数据
        print(f"\n📋 详细数据打印 (前10个episode, 前10个时间步):")
        
        num_episodes_to_show = min(10, qpos_data.shape[0])
        num_timesteps_to_show = min(10, qpos_data.shape[1])
        
        for ep in range(num_episodes_to_show):
            print(f"\n🎯 Episode {ep+1}:")
            print(f"   {'时间步':<8} {'状态数据':<50} {'动作数据':<50}")
            print(f"   {'-'*8} {'-'*50} {'-'*50}")
            
            for t in range(num_timesteps_to_show):
                state = qpos_data[ep, t]
                action = action_data[ep, t]
                
                # 格式化状态和动作数据
                state_str = f"[{', '.join([f'{x:.4f}' for x in state[:5]])}...]"
                action_str = f"[{', '.join([f'{x:.4f}' for x in action[:5]])}...]"
                
                print(f"   {t+1:<8} {state_str:<50} {action_str:<50}")
        
        # 打印完整维度数据
        print(f"\n🔍 完整维度数据 (前5个episode, 前5个时间步):")
        
        for ep in range(min(5, qpos_data.shape[0])):
            print(f"\n📊 Episode {ep+1} 完整数据:")
            
            for t in range(min(5, qpos_data.shape[1])):
                print(f"\n   时间步 {t+1}:")
                
                state = qpos_data[ep, t]
                action = action_data[ep, t]
                
                print(f"     状态: [{', '.join([f'{x:.6f}' for x in state])}]")
                print(f"     动作: [{', '.join([f'{x:.6f}' for x in action])}]")
                
                # 计算差异
                diff = torch.abs(state - action)
                print(f"     差异: [{', '.join([f'{x:.6f}' for x in diff])}]")
        
        # 分析每个维度的变化
        print(f"\n📊 维度分析:")
        print(f"   14个维度的统计信息:")
        
        for dim in range(qpos_data.shape[2]):
            state_dim_data = qpos_data[:, :, dim]
            action_dim_data = action_data[:, :, dim]
            
            print(f"\n   维度 {dim+1}:")
            print(f"     状态 - 范围: [{state_dim_data.min():.6f}, {state_dim_data.max():.6f}], 均值: {state_dim_data.mean():.6f}")
            print(f"     动作 - 范围: [{action_dim_data.min():.6f}, {action_dim_data.max():.6f}], 均值: {action_dim_data.mean():.6f}")
            
            # 检查这个维度是否有变化
            if torch.allclose(state_dim_data, action_dim_data):
                print(f"     ⚠️  该维度状态和动作完全相同")
            else:
                diff = torch.abs(state_dim_data - action_dim_data)
                print(f"     ✅ 该维度有差异 - 最大差异: {diff.max():.6f}, 平均差异: {diff.mean():.6f}")
        
        # 时间序列分析 - 更多episode
        print(f"\n⏰ 时间序列分析 (前5个episode):")
        
        for ep in range(min(5, qpos_data.shape[0])):
            print(f"\n   Episode {ep+1} 时间序列:")
            
            # 选择前3个维度进行时间序列分析
            for dim in range(min(3, qpos_data.shape[2])):
                state_series = qpos_data[ep, :, dim]
                action_series = action_data[ep, :, dim]
                
                print(f"     维度{dim+1}状态变化: [{state_series.min():.6f}, {state_series.max():.6f}]")
                print(f"     维度{dim+1}动作变化: [{action_series.min():.6f}, {action_series.max():.6f}]")
                
                # 显示前15个时间步
                print(f"     前15个时间步状态: {state_series[:15].tolist()}")
                print(f"     前15个时间步动作: {action_series[:15].tolist()}")
                print()
        
        # 添加更多统计信息
        print(f"\n📈 额外统计信息:")
        
        # 分析非零值
        state_nonzero = (qpos_data != 0).float().mean()
        action_nonzero = (action_data != 0).float().mean()
        print(f"   状态数据非零值比例: {state_nonzero:.2%}")
        print(f"   动作数据非零值比例: {action_nonzero:.2%}")
        
        # 分析变化幅度
        state_changes = torch.diff(qpos_data, dim=1)
        action_changes = torch.diff(action_data, dim=1)
        
        print(f"   状态变化幅度 - 范围: [{state_changes.min():.6f}, {state_changes.max():.6f}], 均值: {state_changes.mean():.6f}")
        print(f"   动作变化幅度 - 范围: [{action_changes.min():.6f}, {action_changes.max():.6f}], 均值: {action_changes.mean():.6f}")
        
        # 分析每个episode的运动情况
        print(f"\n🎯 Episode运动分析:")
        
        for ep in range(min(5, qpos_data.shape[0])):
            ep_state = qpos_data[ep]
            ep_action = action_data[ep]
            
            # 计算episode内的变化
            ep_state_changes = torch.diff(ep_state, dim=0)
            ep_action_changes = torch.diff(ep_action, dim=0)
            
            total_movement = torch.norm(ep_state_changes, dim=-1).sum()
            max_movement = torch.norm(ep_state_changes, dim=-1).max()
            
            print(f"   Episode {ep+1}:")
            print(f"     总运动量: {total_movement:.6f}")
            print(f"     最大单步运动: {max_movement:.6f}")
            print(f"     平均单步运动: {torch.norm(ep_state_changes, dim=-1).mean():.6f}")
            
            # 检查是否有运动
            if total_movement > 0.01:
                print(f"     ✅ 有显著运动")
            else:
                print(f"     ⚠️  运动较少")
        
        # 打印一些随机样本
        print(f"\n🎲 随机样本数据:")
        
        import random
        random.seed(42)  # 固定随机种子以便复现
        
        for i in range(3):
            ep = random.randint(0, qpos_data.shape[0]-1)
            t = random.randint(0, qpos_data.shape[1]-1)
            
            state = qpos_data[ep, t]
            action = action_data[ep, t]
            
            print(f"\n   随机样本 {i+1} (Episode {ep+1}, 时间步 {t+1}):")
            print(f"     状态: [{', '.join([f'{x:.6f}' for x in state])}]")
            print(f"     动作: [{', '.join([f'{x:.6f}' for x in action])}]")
            
            diff = torch.abs(state - action)
            print(f"     差异: [{', '.join([f'{x:.6f}' for x in diff])}]")
        
        print(f"\n" + "="*60)
        print("✅ 动作和状态数据打印完成!")
        
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🤖 动作和状态数据打印工具 - 增强版")
    print("="*60)
    
    # 数据路径
    data_path = "/home/xuxuezhou/code/RoboTwin/data/move_can_pot/integration.pkl"
    
    # 打印动作和状态数据
    print_action_state_data(data_path)

if __name__ == "__main__":
    main() 