# Redundancy-Aware Adaptive Layer Pruning Based on a Statistical Framework for Image Classification and Deepfake Detection

## Overview
Deep neural networks often contain **redundant layers or blocks** that contribute little to model performance but increase computational cost.  
This thesis proposes a **statistical framework for adaptive layer pruning**, which automatically identifies and removes redundant layers without manually setting pruning ratios.

- **Key Idea**: Compare feature distributions of adjacent layers using non-parametric statistical tests.  
  - If distributions are not significantly different → the later layer is considered redundant and pruned.  
  - If they are significantly different → the layer is preserved.  
- **Statistical Tests**:  
  - **Mann–Whitney U (MW)**  
  - **Kolmogorov–Smirnov (KS)**  
- **Advantages**:  
  - Data-free pruning (only one forward pass with random input needed)  
  - No need to manually set pruning ratio  
  - Applicable to both **image classification** (e.g., CIFAR-10, ImageNet) and **deepfake detection** (e.g., FaceForensics++, Celeb-DF)

This method achieves a balance between **model compression** and **accuracy preservation**, making it suitable for both efficient deployment and robust deepfake detection.

---

## Quick Start (CIFAR-10 / ResNet56)

### 0. Move into Project Folder

``` bash
cd ./python_resnet_cifar10-master
```

### 1. Train & Evaluate Pretrained Model

``` bash
CUDA_VISIBLE_DEVICES=0 python -u trainer_cifar10.py  --arch=resnet56 --pretrained --evaluate --print-freq=1
```

### 2. Apply Pruning 

``` bash
# Mann–Whitney U test
CUDA_VISIBLE_DEVICES=0 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=MW --save-dir=save_resnet56 |& tee -a log_resnet56_MW

# Kolmogorov–Smirnov test
CUDA_VISIBLE_DEVICES=0 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=KS --save-dir=save_resnet56 |& tee -a log_resnet56_KS
```

### 3. Evaluate Pruned Model

``` bash
# Mann–Whitney U test
CUDA_VISIBLE_DEVICES=0 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=MW --evaluate --print-freq=1

# Kolmogorov–Smirnov test
CUDA_VISIBLE_DEVICES=- python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=KS --evaluate --print-freq=1
```
