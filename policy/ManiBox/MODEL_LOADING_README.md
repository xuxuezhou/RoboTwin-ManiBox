# ManiBox 模型加载配置说明

## 概述

现在支持两种模型加载方式：
1. **自动查找最新模型**：不指定 `ckpt_setting`，自动查找最新的checkpoint
2. **指定具体模型路径**：通过 `ckpt_setting` 参数指定具体的模型路径

## 配置方式

### 1. 自动查找最新模型（原有方式）

在配置文件中不设置 `ckpt_setting` 或设置为 `null`：

```yaml
# Basic experiment configuration
policy_name: ManiBox
task_name: null
task_config: null
ckpt_setting: null  # 或者不设置这一行
seed: null
instruction_type: unseen
```

系统会自动：
- 查找 `ckpt/` 目录下所有以 `2025` 开头的文件夹
- 对于 Diffusion：优先选择包含 `SimpleBBoxDiffusion` 的文件夹
- 对于 RNN：选择最新的文件夹
- 使用找到的最新模型

### 2. 指定具体模型路径（新功能）

在配置文件中设置 `ckpt_setting` 为具体的模型目录名：

```yaml
# Basic experiment configuration
policy_name: ManiBox
task_name: null
task_config: null
ckpt_setting: "2025-07-30_10-47-21SimpleBBoxDiffusion"  # 指定具体的checkpoint目录
seed: null
instruction_type: unseen
```

## 支持的路径格式

### 相对路径（推荐）
```yaml
ckpt_setting: "2025-07-30_10-47-21SimpleBBoxDiffusion"  # 相对于ckpt目录
ckpt_setting: "2025-07-30_10-47-21RNN"                  # 相对于ckpt目录
```

系统会在以下位置查找：
- `policy/ManiBox/ckpt/` (在根目录下运行)
- `ckpt/` (在ManiBox目录下运行)
- 相对于脚本文件的 `ckpt/` 目录

### 绝对路径
```yaml
ckpt_setting: "/home/user/code/RoboTwin/policy/ManiBox/ckpt/2025-07-30_10-47-21SimpleBBoxDiffusion"
```

## 使用示例

### Diffusion 模型指定

1. 创建配置文件 `deploy_policy_diffusion_specific.yml`：
```yaml
# 其他配置保持不变...
ckpt_setting: "2025-07-30_10-47-21SimpleBBoxDiffusion"
policy_class: SimpleBBoxDiffusion
# 其他配置...
```

2. 运行评估：
```bash
bash eval_diffusion.sh pick_diverse_bottles demo_randomized 2025-07-30_10-47-21SimpleBBoxDiffusion 1 0 0
```

### RNN 模型指定

1. 创建配置文件 `deploy_policy_rnn_specific.yml`：
```yaml
# 其他配置保持不变...
ckpt_setting: "2025-07-30_10-47-21RNN"
policy_class: RNN
# 其他配置...
```

2. 运行评估：
```bash
bash eval.sh pick_diverse_bottles demo_randomized 2025-07-30_10-47-21RNN 1 0 0
```

## 错误处理

如果指定的模型路径不存在，系统会给出详细的错误信息：

```
FileNotFoundError: Specified checkpoint '2025-07-30_10-47-21SimpleBBoxDiffusion' not found in any base path: ['policy/ManiBox/ckpt', 'ckpt', '/path/to/script/ckpt']
```

## 调试信息

当使用自动查找模式时，系统会输出详细的调试信息：

```
🔍 No specific checkpoint specified, searching for latest checkpoint...
🔍 Debug: Current working directory: /home/user/code/RoboTwin
🔍 Debug: Script directory: /home/user/code/RoboTwin/policy/ManiBox
🔍 Debug: Checking possible paths:
   1. policy/ManiBox/ckpt - exists: True
      Contents: ['2025-07-30_10-47-21SimpleBBoxDiffusion', '2025-07-30_10-47-21RNN']
🔍 Debug: Found checkpoint directories: ['2025-07-30_10-47-21SimpleBBoxDiffusion', '2025-07-30_10-47-21RNN']
🔍 Debug: SimpleBBoxDiffusion directories: ['2025-07-30_10-47-21SimpleBBoxDiffusion']
🔍 Debug: Selected SimpleBBoxDiffusion checkpoint: 2025-07-30_10-47-21SimpleBBoxDiffusion
📁 Using checkpoint directory: policy/ManiBox/ckpt/2025-07-30_10-47-21SimpleBBoxDiffusion
```

## 注意事项

1. **路径分隔符**：使用正斜杠 `/`，即使在Windows系统上
2. **目录名格式**：checkpoint目录名通常格式为 `YYYY-MM-DD_HH-MM-SS[PolicyType]`
3. **权限问题**：确保有读取checkpoint目录的权限
4. **模型兼容性**：确保指定的模型与配置文件中的 `policy_class` 匹配

## 常见问题

### Q: 如何查看可用的模型？
A: 查看 `ckpt/` 目录下的文件夹：
```bash
ls policy/ManiBox/ckpt/
```

### Q: 如何知道模型类型？
A: 文件夹名通常包含模型类型信息：
- `SimpleBBoxDiffusion` = Diffusion模型
- `RNN` = RNN模型

### Q: 可以同时指定多个模型吗？
A: 不可以，每次只能指定一个模型。如果需要比较多个模型，需要分别运行。 