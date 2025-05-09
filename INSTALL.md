# 🚴‍♂️ Installation
## **Dependencies**

Python versions:

* Python 3.10

Operating systems:

* Linux: Ubuntu 18.04+, Centos 7+


Hardware:

* Rendering: NVIDIA or AMD GPU

* Ray tracing: NVIDIA RTX GPU or AMD equivalent

* Ray-tracing Denoising: NVIDIA GPU

* GPU Simulation: NVIDIA GPU

Software:

* Ray tracing: NVIDIA Driver >= 470
* Denoising (OIDN): NVIDIA Driver >= 520

## 0. Install Vulkan
```
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```

## 1. Basic Env
First, prepare a conda environment.
```bash
conda create -n RoboTwin_Challenge python=3.10
conda activate RoboTwin_Challenge
bash script/_install.sh
```

## 2. Download Assert
```
cd assets
python _download.py
```

## 3. Install GPU-Accelerated Planner (cuRobo)
**cuRobo** is a GPU-accelerated robot motion planning and control library that provides better planning performance. You can install it by following these steps:
```
cd third_party
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../..
```
