# Redundancy-Aware Adaptive Layer Pruning Based on a Statistical Framework for Image Classification and Deepfake Detection


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
