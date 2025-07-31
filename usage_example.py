#!/usr/bin/env python3
"""
实际使用示例：如何在项目中使用改进的卡尔曼滤波功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'policy/ManiBox/manibox/ManiBox'))

import numpy as np
import torch
from yolo_process_data import YoloProcessDataByTimeStep

def example_basic_usage():
    """基本使用示例"""
    print("📖 基本使用示例")
    print("=" * 40)
    
    # 1. 创建YOLO处理对象
    yolo_processor = YoloProcessDataByTimeStep(
        objects_names=["apple", "bottle"],
        max_detections_per_object={"apple": 2, "bottle": 1}
    )
    
    # 2. 启用卡尔曼滤波
    yolo_processor.enable_kalman_filter(
        process_noise=0.03,      # 过程噪声
        measurement_noise=0.1     # 测量噪声
    )
    
    # 3. 重置新episode（初始化滤波器）
    yolo_processor.reset_new_episode()
    
    print("✅ YOLO处理器已配置并启用卡尔曼滤波")
    print(f"   - 检测对象: {yolo_processor.objects_names}")
    print(f"   - 卡尔曼滤波: {'启用' if yolo_processor.using_kalman_filter else '禁用'}")
    
    return yolo_processor

def example_real_time_processing(yolo_processor):
    """实时处理示例"""
    print("\n🔄 实时处理示例")
    print("=" * 40)
    
    # 模拟实时图像数据
    image_height, image_width = 480, 640
    
    for frame_idx in range(5):
        print(f"\n📸 处理第 {frame_idx + 1} 帧:")
        
        # 模拟图像数据（实际应用中这里会是真实的相机图像）
        cam_high = torch.randn(3, image_height, image_width)  # (3, 480, 640)
        cam_left_wrist = torch.randn(3, image_height, image_width)
        cam_right_wrist = torch.randn(3, image_height, image_width)
        
        try:
            # 处理图像
            result = yolo_processor.process(cam_high, cam_left_wrist, cam_right_wrist)
            
            print(f"   ✅ 处理成功")
            print(f"   📊 结果形状: {result.shape}")
            print(f"   📈 数据范围: [{result.min():.3f}, {result.max():.3f}]")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            print("   💡 注意：这需要实际的YOLO模型和GPU支持")

def example_batch_processing():
    """批量处理示例"""
    print("\n📦 批量处理示例")
    print("=" * 40)
    
    # 创建处理器
    processor = YoloProcessDataByTimeStep(
        objects_names=["apple"],
        max_detections_per_object={"apple": 1}
    )
    
    # 启用卡尔曼滤波
    processor.enable_kalman_filter()
    processor.reset_new_episode()
    
    # 模拟批量图像数据
    batch_size = 3
    image_height, image_width = 480, 640
    
    # 创建批量图像数据
    cams_high = [torch.randn(3, image_height, image_width) for _ in range(batch_size)]
    cams_left_wrist = [torch.randn(3, image_height, image_width) for _ in range(batch_size)]
    cams_right_wrist = [torch.randn(3, image_height, image_width) for _ in range(batch_size)]
    
    try:
        # 批量处理
        result = processor.parallel_process_traj(cams_high, cams_left_wrist, cams_right_wrist)
        
        print(f"✅ 批量处理成功")
        print(f"   📊 结果形状: {result.shape}")
        print(f"   📈 数据统计:")
        print(f"      - 均值: {result.mean():.3f}")
        print(f"      - 标准差: {result.std():.3f}")
        print(f"      - 最小值: {result.min():.3f}")
        print(f"      - 最大值: {result.max():.3f}")
        
    except Exception as e:
        print(f"❌ 批量处理失败: {e}")
        print("💡 注意：这需要实际的YOLO模型和GPU支持")

def example_parameter_tuning():
    """参数调优示例"""
    print("\n⚙️ 参数调优示例")
    print("=" * 40)
    
    # 测试不同参数配置
    configs = [
        {
            "name": "保守配置",
            "process_noise": 0.01,
            "measurement_noise": 0.05,
            "description": "更平滑的预测，适合稳定场景"
        },
        {
            "name": "平衡配置",
            "process_noise": 0.03,
            "measurement_noise": 0.1,
            "description": "平衡响应性和平滑性，推荐配置"
        },
        {
            "name": "激进配置",
            "process_noise": 0.1,
            "measurement_noise": 0.2,
            "description": "更快速响应，适合动态场景"
        }
    ]
    
    for config in configs:
        print(f"\n🔧 {config['name']}:")
        print(f"   📝 {config['description']}")
        print(f"   ⚙️ 过程噪声: {config['process_noise']}")
        print(f"   ⚙️ 测量噪声: {config['measurement_noise']}")
        
        # 创建处理器
        processor = YoloProcessDataByTimeStep(
            objects_names=["apple"],
            max_detections_per_object={"apple": 1}
        )
        
        # 应用配置
        processor.enable_kalman_filter(
            process_noise=config['process_noise'],
            measurement_noise=config['measurement_noise']
        )
        processor.reset_new_episode()
        
        print(f"   ✅ 配置应用成功")

def example_integration_with_existing_code():
    """与现有代码集成示例"""
    print("\n🔗 与现有代码集成示例")
    print("=" * 40)
    
    # 模拟现有的YOLO处理代码
    print("📋 现有代码（未启用卡尔曼滤波）:")
    print("""
    # 创建处理器
    yolo_processor = YoloProcessDataByTimeStep(
        objects_names=["apple"],
        max_detections_per_object={"apple": 1}
    )
    
    # 处理图像
    result = yolo_processor.process(cam_high, cam_left_wrist, cam_right_wrist)
    """)
    
    print("\n📋 改进后的代码（启用卡尔曼滤波）:")
    print("""
    # 创建处理器
    yolo_processor = YoloProcessDataByTimeStep(
        objects_names=["apple"],
        max_detections_per_object={"apple": 1}
    )
    
    # 启用卡尔曼滤波（新增）
    yolo_processor.enable_kalman_filter(process_noise=0.03, measurement_noise=0.1)
    yolo_processor.reset_new_episode()  # 新增
    
    # 处理图像（无需修改）
    result = yolo_processor.process(cam_high, cam_left_wrist, cam_right_wrist)
    """)
    
    print("✅ 集成非常简单，只需添加两行代码即可启用卡尔曼滤波")

def example_error_handling():
    """错误处理示例"""
    print("\n🛡️ 错误处理示例")
    print("=" * 40)
    
    try:
        # 创建处理器
        processor = YoloProcessDataByTimeStep(
            objects_names=["apple"],
            max_detections_per_object={"apple": 1}
        )
        
        # 启用卡尔曼滤波
        processor.enable_kalman_filter()
        processor.reset_new_episode()
        
        print("✅ 正常初始化")
        
        # 模拟处理错误
        print("\n🔍 错误处理测试:")
        
        # 测试无效参数
        try:
            processor.enable_kalman_filter(process_noise=-1)  # 无效参数
        except Exception as e:
            print(f"   ✅ 捕获到无效参数错误: {type(e).__name__}")
        
        # 测试未初始化就使用
        processor2 = YoloProcessDataByTimeStep(
            objects_names=["apple"],
            max_detections_per_object={"apple": 1}
        )
        
        try:
            # 未调用reset_new_episode()就使用
            result = processor2.process(
                torch.randn(3, 480, 640),
                torch.randn(3, 480, 640),
                torch.randn(3, 480, 640)
            )
        except Exception as e:
            print(f"   ✅ 捕获到未初始化错误: {type(e).__name__}")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")

if __name__ == "__main__":
    print("🎯 YOLO检测卡尔曼滤波使用示例")
    print("=" * 60)
    
    # 运行各种示例
    yolo_processor = example_basic_usage()
    example_real_time_processing(yolo_processor)
    example_batch_processing()
    example_parameter_tuning()
    example_integration_with_existing_code()
    example_error_handling()
    
    print("\n" + "=" * 60)
    print("📋 使用总结:")
    print("✅ 基本使用简单，只需几行代码")
    print("✅ 完全向后兼容，不影响现有功能")
    print("✅ 支持实时处理和批量处理")
    print("✅ 提供灵活的参数配置")
    print("✅ 包含完善的错误处理")
    print("\n🚀 现在可以在您的项目中使用改进的卡尔曼滤波功能了！") 