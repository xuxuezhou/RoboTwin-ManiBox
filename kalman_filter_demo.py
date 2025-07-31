#!/usr/bin/env python3
"""
卡尔曼滤波演示脚本
展示如何使用改进的卡尔曼滤波来处理YOLO检测失败的情况
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'policy/ManiBox/manibox/ManiBox'))

import numpy as np
import cv2
import torch
from yolo_process_data import YoloProcessDataByTimeStep, KalmanFilter

def demo_kalman_filter():
    """演示卡尔曼滤波的使用"""
    print("🚀 卡尔曼滤波演示")
    print("=" * 50)
    
    # 创建YOLO处理对象
    yolo_processor = YoloProcessDataByTimeStep(
        objects_names=["apple"],
        max_detections_per_object={"apple": 1}
    )
    
    # 启用卡尔曼滤波
    yolo_processor.enable_kalman_filter(process_noise=0.03, measurement_noise=0.1)
    
    # 重置新episode（创建卡尔曼滤波器）
    yolo_processor.reset_new_episode()
    
    # 模拟检测结果序列
    # 格式: [x, y, width, height] 或 KalmanFilter.NO_BBOX
    detection_sequence = [
        [0.3, 0.4, 0.1, 0.15],  # 第1帧：检测成功
        KalmanFilter.NO_BBOX,     # 第2帧：检测失败
        KalmanFilter.NO_BBOX,     # 第3帧：检测失败
        [0.35, 0.45, 0.12, 0.16], # 第4帧：检测成功
        KalmanFilter.NO_BBOX,     # 第5帧：检测失败
        [0.4, 0.5, 0.11, 0.14],  # 第6帧：检测成功
    ]
    
    print("📊 检测序列演示:")
    print("帧数 | 原始检测 | 卡尔曼滤波结果")
    print("-" * 40)
    
    # 创建单个卡尔曼滤波器进行演示
    kalman_filter = KalmanFilter(process_noise=0.03, measurement_noise=0.1)
    
    for i, detection in enumerate(detection_sequence):
        # 应用卡尔曼滤波
        filtered_detection = kalman_filter.fill_missing_bbox_with_kalman(detection)
        
        # 格式化输出
        if detection is KalmanFilter.NO_BBOX:
            original_str = "检测失败"
        else:
            original_str = f"[{detection[0]:.3f}, {detection[1]:.3f}, {detection[2]:.3f}, {detection[3]:.3f}]"
        
        filtered_str = f"[{filtered_detection[0]:.3f}, {filtered_detection[1]:.3f}, {filtered_detection[2]:.3f}, {filtered_detection[3]:.3f}]"
        
        print(f"  {i+1:2d} | {original_str:>12} | {filtered_str}")
    
    print("\n" + "=" * 50)
    print("✅ 演示完成！")
    
    # 显示滤波器状态信息
    print(f"\n📈 滤波器状态:")
    print(f"   - 已初始化: {kalman_filter.is_initialized}")
    print(f"   - 连续失败次数: {kalman_filter.consecutive_failures}")
    print(f"   - 最大连续失败次数: {kalman_filter.max_consecutive_failures}")

def demo_batch_processing():
    """演示批量处理"""
    print("\n🔄 批量处理演示")
    print("=" * 50)
    
    # 创建YOLO处理对象
    yolo_processor = YoloProcessDataByTimeStep(
        objects_names=["apple"],
        max_detections_per_object={"apple": 1}
    )
    
    # 启用卡尔曼滤波
    yolo_processor.enable_kalman_filter()
    yolo_processor.reset_new_episode()
    
    # 模拟多帧图像数据
    batch_size = 5
    image_height, image_width = 480, 640
    
    # 创建模拟图像数据
    cam_high = [np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8) for _ in range(batch_size)]
    cam_left_wrist = [np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8) for _ in range(batch_size)]
    cam_right_wrist = [np.random.randint(0, 255, (image_height, image_width, 3), dtype=np.uint8) for _ in range(batch_size)]
    
    print(f"📸 处理 {batch_size} 帧图像...")
    
    try:
        # 处理图像序列
        result = yolo_processor.parallel_process_traj(cam_high, cam_left_wrist, cam_right_wrist)
        print(f"✅ 批量处理完成！结果形状: {result.shape}")
        print(f"   结果类型: {type(result)}")
        print(f"   设备: {result.device if hasattr(result, 'device') else 'CPU'}")
        
    except Exception as e:
        print(f"❌ 批量处理失败: {e}")
        print("   注意：这可能需要实际的YOLO模型和GPU支持")

def demo_parameter_tuning():
    """演示参数调优"""
    print("\n⚙️ 参数调优演示")
    print("=" * 50)
    
    # 测试不同的噪声参数
    test_cases = [
        (0.01, 0.05, "低噪声 - 更平滑但可能滞后"),
        (0.03, 0.1, "中等噪声 - 平衡响应性和平滑性"),
        (0.1, 0.2, "高噪声 - 更快速响应但可能不稳定"),
    ]
    
    # 模拟检测序列
    test_sequence = [
        [0.3, 0.4, 0.1, 0.15],
        KalmanFilter.NO_BBOX,
        KalmanFilter.NO_BBOX,
        [0.35, 0.45, 0.12, 0.16],
        KalmanFilter.NO_BBOX,
    ]
    
    for process_noise, measurement_noise, description in test_cases:
        print(f"\n🔧 {description}")
        print(f"   过程噪声: {process_noise}, 测量噪声: {measurement_noise}")
        
        kalman_filter = KalmanFilter(process_noise, measurement_noise)
        
        print("   帧数 | 原始检测 | 滤波结果")
        print("   " + "-" * 35)
        
        for i, detection in enumerate(test_sequence):
            filtered = kalman_filter.fill_missing_bbox_with_kalman(detection)
            
            if detection is KalmanFilter.NO_BBOX:
                original_str = "检测失败"
            else:
                original_str = f"[{detection[0]:.3f}, {detection[1]:.3f}]"
            
            filtered_str = f"[{filtered[0]:.3f}, {filtered[1]:.3f}]"
            print(f"     {i+1:2d} | {original_str:>10} | {filtered_str}")

if __name__ == "__main__":
    print("🎯 YOLO检测卡尔曼滤波改进演示")
    print("=" * 60)
    
    # 运行演示
    demo_kalman_filter()
    demo_parameter_tuning()
    
    print("\n" + "=" * 60)
    print("📋 使用说明:")
    print("1. 创建 YoloProcessDataByTimeStep 对象")
    print("2. 调用 enable_kalman_filter() 启用卡尔曼滤波")
    print("3. 调用 reset_new_episode() 初始化滤波器")
    print("4. 正常使用 process() 或 parallel_process_traj() 方法")
    print("5. 卡尔曼滤波会自动处理检测失败的情况")
    
    print("\n🔧 主要改进:")
    print("✅ 8维状态向量（位置+速度+尺寸+速度）")
    print("✅ 自适应噪声参数")
    print("✅ 连续失败次数限制")
    print("✅ 边界框合理性检查")
    print("✅ 图像范围约束")
    print("✅ 更好的初始化机制") 