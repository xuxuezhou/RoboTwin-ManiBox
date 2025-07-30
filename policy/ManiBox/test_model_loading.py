#!/usr/bin/env python3
"""
测试模型加载功能
"""

import os
import sys
import yaml

# 添加路径
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

def test_diffusion_model_loading():
    """测试Diffusion模型加载"""
    print("🧪 Testing Diffusion model loading...")
    
    # 测试自动查找
    print("\n1. Testing auto-find latest model:")
    try:
        from deploy_policy_diffusion import get_model
        import argparse
        
        # 创建测试参数
        args = argparse.Namespace()
        args.config = "deploy_policy_diffusion.yml"
        args.overrides = []
        
        # 测试自动查找
        model = get_model(args)
        print("✅ Auto-find successful")
        
    except Exception as e:
        print(f"❌ Auto-find failed: {e}")
    
    # 测试指定模型
    print("\n2. Testing specific model loading:")
    try:
        # 创建指定模型的配置
        config = {
            'ckpt_setting': '2025-07-30_00-18-14SimpleBBoxDiffusion',
            'policy_class': 'SimpleBBoxDiffusion',
            'task_name': 'pick_diverse_bottles'
        }
        
        # 保存临时配置文件
        temp_config_path = 'temp_diffusion_config.yml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # 测试指定模型加载
        args.config = temp_config_path
        model = get_model(args)
        print("✅ Specific model loading successful")
        
        # 清理临时文件
        os.remove(temp_config_path)
        
    except Exception as e:
        print(f"❌ Specific model loading failed: {e}")
        if os.path.exists('temp_diffusion_config.yml'):
            os.remove('temp_diffusion_config.yml')

def test_rnn_model_loading():
    """测试RNN模型加载"""
    print("\n🧪 Testing RNN model loading...")
    
    # 测试自动查找
    print("\n1. Testing auto-find latest model:")
    try:
        from deploy_policy import get_model
        import argparse
        
        # 创建测试参数
        args = argparse.Namespace()
        args.config = "deploy_policy.yml"
        args.overrides = []
        
        # 测试自动查找
        model = get_model(args)
        print("✅ Auto-find successful")
        
    except Exception as e:
        print(f"❌ Auto-find failed: {e}")
    
    # 测试指定模型
    print("\n2. Testing specific model loading:")
    try:
        # 创建指定模型的配置
        config = {
            'ckpt_setting': '2025-07-30_00-18-14RNN',  # 假设有这个RNN模型
            'policy_class': 'RNN',
            'task_name': 'pick_diverse_bottles'
        }
        
        # 保存临时配置文件
        temp_config_path = 'temp_rnn_config.yml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # 测试指定模型加载
        args.config = temp_config_path
        model = get_model(args)
        print("✅ Specific model loading successful")
        
        # 清理临时文件
        os.remove(temp_config_path)
        
    except Exception as e:
        print(f"❌ Specific model loading failed: {e}")
        if os.path.exists('temp_rnn_config.yml'):
            os.remove('temp_rnn_config.yml')

def main():
    """主函数"""
    print("🚀 Starting model loading tests...")
    
    # 检查当前目录
    print(f"📁 Current directory: {os.getcwd()}")
    
    # 检查ckpt目录
    ckpt_dir = "ckpt"
    if os.path.exists(ckpt_dir):
        print(f"📁 Found ckpt directory: {ckpt_dir}")
        ckpt_contents = os.listdir(ckpt_dir)
        print(f"   Contents: {ckpt_contents}")
    else:
        print(f"❌ Ckpt directory not found: {ckpt_dir}")
        return
    
    # 测试Diffusion模型加载
    test_diffusion_model_loading()
    
    # 测试RNN模型加载
    test_rnn_model_loading()
    
    print("\n🎉 Model loading tests completed!")

if __name__ == "__main__":
    main() 