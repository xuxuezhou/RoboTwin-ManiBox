#!/usr/bin/env python3
"""
专门分析integration.pkl中图像数据结构的脚本
"""

import os
import torch
import numpy as np

def analyze_image_data_structure(data_path):
    """
    分析图像数据的详细结构
    
    Args:
        data_path: integration.pkl文件的路径
    """
    print("🔍 分析图像数据结构...")
    print(f"📁 数据路径: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ 错误: 文件不存在 - {data_path}")
        return
    
    try:
        # 加载数据
        with open(data_path, 'rb') as f:
            data = torch.load(f, map_location='cpu')
        
        image_data = data['image_data']
        print("✅ 图像数据加载成功!")
        print("\n" + "="*60)
        
        # 基本信息
        print("📊 图像数据基本信息:")
        print(f"   形状: {image_data.shape}")
        print(f"   数据类型: {image_data.dtype}")
        print(f"   Episode数量: {image_data.shape[0]}")
        print(f"   时间步数量: {image_data.shape[1]}")
        print(f"   特征维度: {image_data.shape[2]}")
        
        # 分析特征维度结构
        print(f"\n🔍 特征维度分析 (24维):")
        print("   根据process_data.py，这24维数据包含:")
        print("   - 3个相机 (head_camera, left_camera, right_camera)")
        print("   - 每个相机2个检测 (max_detections_per_object=2)")
        print("   - 每个检测4个坐标 (x1, y1, x2, y2) - xyxyn格式")
        print("   - 总计: 3 × 2 × 4 = 24维")
        
        # 分析每个相机的数据
        print(f"\n📷 相机数据分布:")
        cameras = ["head_camera", "left_camera", "right_camera"]
        detections_per_camera = 2
        coords_per_detection = 4
        
        for cam_idx, cam_name in enumerate(cameras):
            start_idx = cam_idx * detections_per_camera * coords_per_detection
            end_idx = start_idx + detections_per_camera * coords_per_detection
            
            print(f"\n   {cam_name}:")
            print(f"     索引范围: {start_idx} - {end_idx-1}")
            print(f"     数据维度: {detections_per_camera} 个检测 × {coords_per_detection} 个坐标")
            
            # 分析这个相机的数据
            cam_data = image_data[:, :, start_idx:end_idx]
            print(f"     数据形状: {cam_data.shape}")
            print(f"     数值范围: [{cam_data.min():.6f}, {cam_data.max():.6f}]")
            print(f"     非零值比例: {(cam_data != 0).float().mean():.2%}")
            
            # 显示一些样本
            print(f"     样本数据 (Episode 1, 时间步 1):")
            sample = cam_data[0, 0].reshape(detections_per_camera, coords_per_detection)
            for det_idx in range(detections_per_camera):
                bbox = sample[det_idx]
                if torch.all(bbox == 0):
                    print(f"        检测 {det_idx+1}: [0, 0, 0, 0] (无检测)")
                else:
                    print(f"        检测 {det_idx+1}: [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")
        
        # 分析检测质量
        print(f"\n📈 检测质量分析:")
        
        # 统计有效检测数量
        total_detections = 0
        valid_detections = 0
        
        for ep in range(min(5, image_data.shape[0])):  # 分析前5个episode
            for t in range(min(10, image_data.shape[1])):  # 分析前10个时间步
                for cam_idx in range(3):
                    start_idx = cam_idx * 8
                    end_idx = start_idx + 8
                    cam_data = image_data[ep, t, start_idx:end_idx].reshape(2, 4)
                    
                    for det_idx in range(2):
                        bbox = cam_data[det_idx]
                        total_detections += 1
                        if not torch.all(bbox == 0):
                            valid_detections += 1
        
        detection_rate = valid_detections / total_detections if total_detections > 0 else 0
        print(f"   有效检测率: {detection_rate:.2%}")
        print(f"   总检测次数: {total_detections}")
        print(f"   有效检测次数: {valid_detections}")
        
        # 分析边界框坐标分布
        print(f"\n📐 边界框坐标分析:")
        
        # 收集所有非零边界框
        valid_bboxes = []
        for ep in range(image_data.shape[0]):
            for t in range(image_data.shape[1]):
                for cam_idx in range(3):
                    start_idx = cam_idx * 8
                    end_idx = start_idx + 8
                    cam_data = image_data[ep, t, start_idx:end_idx].reshape(2, 4)
                    
                    for det_idx in range(2):
                        bbox = cam_data[det_idx]
                        if not torch.all(bbox == 0):
                            valid_bboxes.append(bbox.numpy())
        
        if valid_bboxes:
            valid_bboxes = np.array(valid_bboxes)
            print(f"   有效边界框数量: {len(valid_bboxes)}")
            print(f"   x1 范围: [{valid_bboxes[:, 0].min():.4f}, {valid_bboxes[:, 0].max():.4f}]")
            print(f"   y1 范围: [{valid_bboxes[:, 1].min():.4f}, {valid_bboxes[:, 1].max():.4f}]")
            print(f"   x2 范围: [{valid_bboxes[:, 2].min():.4f}, {valid_bboxes[:, 2].max():.4f}]")
            print(f"   y2 范围: [{valid_bboxes[:, 3].min():.4f}, {valid_bboxes[:, 3].max():.4f}]")
            
            # 检查边界框合理性
            invalid_bboxes = 0
            for bbox in valid_bboxes:
                x1, y1, x2, y2 = bbox
                if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                    invalid_bboxes += 1
            
            print(f"   不合理边界框数量: {invalid_bboxes}")
            print(f"   不合理比例: {invalid_bboxes/len(valid_bboxes):.2%}")
        
        # 时间序列分析
        print(f"\n⏰ 时间序列分析:")
        
        # 分析检测连续性
        continuity_analysis = []
        for ep in range(min(3, image_data.shape[0])):
            ep_continuity = []
            for cam_idx in range(3):
                start_idx = cam_idx * 8
                end_idx = start_idx + 8
                cam_data = image_data[ep, :, start_idx:end_idx].reshape(-1, 2, 4)
                
                # 检查每个检测槽位的连续性
                for det_idx in range(2):
                    det_series = cam_data[:, det_idx]
                    valid_frames = torch.any(det_series != 0, dim=1)
                    continuity = valid_frames.float().mean()
                    ep_continuity.append(continuity.item())
            
            continuity_analysis.append(ep_continuity)
            print(f"   Episode {ep+1} 检测连续性:")
            for cam_idx, cam_name in enumerate(cameras):
                for det_idx in range(2):
                    continuity = ep_continuity[cam_idx * 2 + det_idx]
                    print(f"     {cam_name} 检测{det_idx+1}: {continuity:.2%}")
        
        print(f"\n" + "="*60)
        print("✅ 图像数据结构分析完成!")
        
    except Exception as e:
        print(f"❌ 分析数据时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🎯 图像数据结构分析工具")
    print("="*60)
    
    # 数据路径
    data_path = "/home/xuxuezhou/code/RoboTwin/data/move_can_pot/integration.pkl"
    
    # 分析图像数据
    analyze_image_data_structure(data_path)

if __name__ == "__main__":
    main() 