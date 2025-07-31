#!/usr/bin/env python3
"""
测试卡尔曼滤波在实际YOLO处理中的集成效果
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'policy/ManiBox/manibox/ManiBox'))

import numpy as np
import torch
from yolo_process_data import YoloProcessDataByTimeStep, KalmanFilter

def test_kalman_integration():
    """测试卡尔曼滤波集成效果"""
    print("🧪 测试卡尔曼滤波集成效果")
    print("=" * 50)
    
    # 测试1: 基本功能测试
    print("📋 测试1: 基本功能测试")
    
    # 创建处理器
    processor = YoloProcessDataByTimeStep(
        objects_names=["apple"],
        max_detections_per_object={"apple": 1}
    )
    
    # 测试默认状态（未启用卡尔曼滤波）
    assert processor.using_kalman_filter == False, "默认应该禁用卡尔曼滤波"
    print("✅ 默认状态正确")
    
    # 启用卡尔曼滤波
    processor.enable_kalman_filter(process_noise=0.03, measurement_noise=0.1)
    assert processor.using_kalman_filter == True, "启用后应该启用卡尔曼滤波"
    print("✅ 启用功能正常")
    
    # 重置episode
    processor.reset_new_episode()
    assert hasattr(processor, 'kalman_filter_objects'), "应该创建卡尔曼滤波器对象"
    print("✅ 重置功能正常")
    
    # 测试2: 卡尔曼滤波器状态测试
    print("\n📋 测试2: 卡尔曼滤波器状态测试")
    
    kalman_filter = KalmanFilter(process_noise=0.03, measurement_noise=0.1)
    
    # 测试初始状态
    assert kalman_filter.is_initialized == False, "初始状态应该未初始化"
    assert kalman_filter.consecutive_failures == 0, "初始失败次数应该为0"
    print("✅ 初始状态正确")
    
    # 测试首次检测
    first_detection = [0.3, 0.4, 0.1, 0.15]
    result = kalman_filter.fill_missing_bbox_with_kalman(first_detection)
    assert kalman_filter.is_initialized == True, "首次检测后应该已初始化"
    assert result == first_detection, "首次检测应该直接返回原值"
    print("✅ 首次检测处理正确")
    
    # 测试检测失败处理
    failure_result = kalman_filter.fill_missing_bbox_with_kalman(KalmanFilter.NO_BBOX)
    assert failure_result != KalmanFilter.NO_BBOX, "应该返回预测值而不是NO_BBOX"
    assert kalman_filter.consecutive_failures == 1, "失败次数应该增加"
    print("✅ 检测失败处理正确")
    
    # 测试3: 参数调优测试
    print("\n📋 测试3: 参数调优测试")
    
    # 测试不同噪声参数
    test_params = [
        (0.01, 0.05, "低噪声"),
        (0.03, 0.1, "中等噪声"),
        (0.1, 0.2, "高噪声")
    ]
    
    for process_noise, measurement_noise, description in test_params:
        kalman = KalmanFilter(process_noise, measurement_noise)
        
        # 模拟检测序列
        detections = [
            [0.3, 0.4, 0.1, 0.15],
            KalmanFilter.NO_BBOX,
            [0.35, 0.45, 0.12, 0.16]
        ]
        
        results = []
        for det in detections:
            result = kalman.fill_missing_bbox_with_kalman(det)
            results.append(result)
        
        # 验证结果合理性
        assert len(results) == 3, f"{description}: 结果数量应该为3"
        assert results[0] == detections[0], f"{description}: 首次检测应该保持不变"
        assert results[2] == detections[2], f"{description}: 成功检测应该保持不变"
        
        print(f"✅ {description} 参数测试通过")
    
    # 测试4: 边界条件测试
    print("\n📋 测试4: 边界条件测试")
    
    kalman = KalmanFilter()
    
    # 先给滤波器一个成功的检测来初始化它
    kalman.fill_missing_bbox_with_kalman([0.3, 0.4, 0.1, 0.15])
    
    # 测试连续多次失败
    for i in range(15):  # 超过最大失败次数
        result = kalman.fill_missing_bbox_with_kalman(KalmanFilter.NO_BBOX)
        if i < 10:  # 前10次应该返回预测值
            assert result != KalmanFilter.NO_BBOX, f"第{i+1}次失败应该返回预测值"
        else:  # 超过限制后应该返回NO_BBOX
            assert result == KalmanFilter.NO_BBOX, f"第{i+1}次失败应该返回NO_BBOX"
    
    print("✅ 连续失败限制测试通过")
    
    # 测试重置功能
    kalman.reset()
    assert kalman.is_initialized == False, "重置后应该未初始化"
    assert kalman.consecutive_failures == 0, "重置后失败次数应该为0"
    print("✅ 重置功能测试通过")
    
    # 测试5: 批量处理测试
    print("\n📋 测试5: 批量处理测试")
    
    processor = YoloProcessDataByTimeStep(
        objects_names=["apple"],
        max_detections_per_object={"apple": 1}
    )
    processor.enable_kalman_filter()
    processor.reset_new_episode()
    
    # 验证滤波器对象创建
    assert len(processor.kalman_filter_objects) == 3, "应该有3个相机的滤波器"
    assert len(processor.kalman_filter_objects[0]) == 1, "每个相机应该有1个对象类型"
    assert len(processor.kalman_filter_objects[0][0]) == 1, "每个对象类型应该有1个检测槽位"
    
    print("✅ 批量处理滤波器创建正确")
    
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！")
    print("\n📊 测试总结:")
    print("✅ 基本功能测试通过")
    print("✅ 状态管理测试通过")
    print("✅ 参数调优测试通过")
    print("✅ 边界条件测试通过")
    print("✅ 批量处理测试通过")

def test_performance_comparison():
    """性能对比测试"""
    print("\n⚡ 性能对比测试")
    print("=" * 50)
    
    # 创建测试数据
    test_sequence = [
        [0.3, 0.4, 0.1, 0.15],
        KalmanFilter.NO_BBOX,
        KalmanFilter.NO_BBOX,
        [0.35, 0.45, 0.12, 0.16],
        KalmanFilter.NO_BBOX,
        [0.4, 0.5, 0.11, 0.14],
        KalmanFilter.NO_BBOX,
        KalmanFilter.NO_BBOX,
        [0.42, 0.52, 0.13, 0.15]
    ]
    
    # 测试不同配置的性能
    configs = [
        ("低噪声", 0.01, 0.05),
        ("中等噪声", 0.03, 0.1),
        ("高噪声", 0.1, 0.2)
    ]
    
    for name, process_noise, measurement_noise in configs:
        print(f"\n🔧 {name} 配置测试:")
        print(f"   过程噪声: {process_noise}, 测量噪声: {measurement_noise}")
        
        kalman = KalmanFilter(process_noise, measurement_noise)
        
        print("   帧数 | 原始检测 | 滤波结果")
        print("   " + "-" * 35)
        
        for i, detection in enumerate(test_sequence):
            filtered = kalman.fill_missing_bbox_with_kalman(detection)
            
            if detection is KalmanFilter.NO_BBOX:
                original_str = "检测失败"
            else:
                original_str = f"[{detection[0]:.3f}, {detection[1]:.3f}]"
            
            filtered_str = f"[{filtered[0]:.3f}, {filtered[1]:.3f}]"
            print(f"     {i+1:2d} | {original_str:>10} | {filtered_str}")
        
        print(f"   ✅ {name} 配置测试完成")

if __name__ == "__main__":
    print("🧪 卡尔曼滤波集成测试")
    print("=" * 60)
    
    # 运行功能测试
    test_kalman_integration()
    
    # 运行性能对比测试
    test_performance_comparison()
    
    print("\n" + "=" * 60)
    print("📋 测试完成总结:")
    print("✅ 所有功能测试通过")
    print("✅ 性能对比测试完成")
    print("✅ 卡尔曼滤波集成成功")
    print("\n🚀 现在可以在实际项目中使用改进的卡尔曼滤波功能了！") 