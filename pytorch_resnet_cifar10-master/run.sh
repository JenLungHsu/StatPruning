#!/bin/bash

for model in resnet20 resnet32 resnet44 resnet56 resnet110 resnet1202
do
    echo "python -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
    python -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model
done

# ---------------------------------------------------------------------------------------------------------------------------------

### CIFAR10

### R56
# pretrained model
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56 --pretrained --evaluate --print-freq=1

# pruning method 
CUDA_VISIBLE_DEVICES=2 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=MW --save-dir=save_resnet56 |& tee -a log_resnet56_MW
CUDA_VISIBLE_DEVICES=2 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=KS --save-dir=save_resnet56 |& tee -a log_resnet56_KS
CUDA_VISIBLE_DEVICES=5 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.025 --save-dir=save_resnet56_sub |& tee -a log_resnet56_emd0.025_sub
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.05 --save-dir=save_resnet56 |& tee -a log_resnet56_emd0.05
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.075 --save-dir=save_resnet56 |& tee -a log_resnet56_emd0.075
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.1 --save-dir=save_resnet56 |& tee -a log_resnet56_emd0.1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=RedCircle --save-dir=save_resnet56 |& tee -a log_resnet56_RedCircle
CUDA_VISIBLE_DEVICES=5 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.025_dr --save-dir=save_resnet56_sub |& tee -a log_resnet56_emd0.025_dr_sub
CUDA_VISIBLE_DEVICES=5 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.05_dr --save-dir=save_resnet56_sub |& tee -a log_resnet56_emd0.05_dr_sub

# evaluate 
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=MW --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=KS --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=5 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.025 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.05 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.075 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.1 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=RedCircle --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=5 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.025_dr --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10.py  --arch=resnet56  --pruning-method=EMD0.05_dr --evaluate --print-freq=1

### R110
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10_resnet110.py  --arch=resnet110 --pretrained --evaluate --print-freq=1

CUDA_VISIBLE_DEVICES=2 python -u trainer_cifar10_resnet110.py  --arch=resnet110  --pruning-method=MW --save-dir=save_resnet110 |& tee -a log_resnet110_MW
CUDA_VISIBLE_DEVICES=2 python -u trainer_cifar10_resnet110.py  --arch=resnet110  --pruning-method=KS --save-dir=save_resnet110 |& tee -a log_resnet110_KS
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar10_resnet110.py  --arch=resnet110  --pruning-method=KS --save-dir=save_resnet110_KS_e300 |& tee -a log_resnet110_KS_e300
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar10_resnet110.py  --arch=resnet110  --pruning-method=KS_seed5 --save-dir=save_resnet110_KS_seed5 |& tee -a log_resnet110_KS_seed5

CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10_resnet110.py  --arch=resnet110  --pruning-method=MW --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar10_resnet110.py  --arch=resnet110  --pruning-method=KS --evaluate --print-freq=1

### R164 
# pretrained model
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar_resnet164.py  --arch=resnet164_c10 --pretrained --evaluate --print-freq=1

# pruning method 
CUDA_VISIBLE_DEVICES=1 python -u trainer_cifar_resnet164.py  --arch=resnet164_c10  --pruning-method=MW --save-dir=save_resnet164_c10 |& tee -a log_resnet164_c10_MW
CUDA_VISIBLE_DEVICES=1 python -u trainer_cifar_resnet164.py  --arch=resnet164_c10  --pruning-method=KS --save-dir=save_resnet164_c10 |& tee -a log_resnet164_c10_KS

# evaluate 
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar_resnet164.py  --arch=resnet164_c10  --pruning-method=MW --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=4 python -u trainer_cifar_resnet164.py  --arch=resnet164_c10  --pruning-method=KS --evaluate --print-freq=1
# ---------------------------------------------------------------------------------------------------------------------------------

### CIFAR100

## R56
# pretrained model
CUDA_VISIBLE_DEVICES=4 python -u trainer_cifar100.py  --arch=resnet56 --save-dir=save_resnet56_cifar100 |& tee -a log_resnet56_cifar100
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56 --pretrained --evaluate --print-freq=1

# pruning method 
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=MW --save-dir=save_resnet56_cifar100 |& tee -a log_resnet56_cifar100_MW
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=KS --save-dir=save_resnet56_cifar100 |& tee -a log_resnet56_cifar100_KS
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=EMD0.025 --save-dir=save_resnet56_cifar100_sub |& tee -a log_resnet56_cifar100_emd0.025_sub
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=EMD0.05 --save-dir=save_resnet56_cifar100 |& tee -a log_resnet56_cifar100_emd0.05
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=EMD0.075 --save-dir=save_resnet56_cifar100 |& tee -a log_resnet56_cifar100_emd0.075
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=EMD0.1 --save-dir=save_resnet56_cifar100 |& tee -a log_resnet56_cifar100_emd0.1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=RedCircle --save-dir=save_resnet56_cifar100 |& tee -a log_resnet56_cifar100_RedCircle

# evaluate 
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=MW --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=KS --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=5 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=EMD0.025 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=EMD0.05 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=EMD0.075 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=EMD0.1 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100.py  --arch=resnet56  --pruning-method=RedCircle --evaluate --print-freq=1

## R110
# pretrained model
CUDA_VISIBLE_DEVICES=2 python -u trainer_cifar100_resnet110.py  --arch=resnet110 --save-dir=save_resnet110_cifar100 |& tee -a log_resnet110_cifar100
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100_resnet110.py  --arch=resnet110 --pretrained --evaluate --print-freq=1

# pruning method 
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar100_resnet110.py  --arch=resnet110  --pruning-method=MW --save-dir=save_resnet110_cifar100 |& tee -a log_resnet110_cifar100_MW
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar100_resnet110.py  --arch=resnet110  --pruning-method=KS --save-dir=save_resnet110_cifar100 |& tee -a log_resnet110_cifar100_KS
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar100_resnet110.py  --arch=resnet110  --pruning-method=KS_remove --save-dir=save_resnet110_cifar100 |& tee -a log_resnet110_cifar100_KS_remove

# evaluate 
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100_resnet110.py  --arch=resnet110  --pruning-method=MW --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100_resnet110.py  --arch=resnet110  --pruning-method=KS --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=6 python -u trainer_cifar100_resnet110.py  --arch=resnet110  --pruning-method=KS_remove --evaluate --print-freq=1

### R164 
# pretrained model
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar_resnet164.py  --arch=resnet164_c100 --save-dir=save_resnet164_c100 |& tee -a log_resnet164_c100
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar_resnet164.py  --arch=resnet164_c100 --pretrained --evaluate --print-freq=1

# pruning method 
CUDA_VISIBLE_DEVICES=1 python -u trainer_cifar_resnet164.py  --arch=resnet164_c100  --pruning-method=MW --save-dir=save_resnet164_c100 |& tee -a log_resnet164_c100_MW
CUDA_VISIBLE_DEVICES=1 python -u trainer_cifar_resnet164.py  --arch=resnet164_c100  --pruning-method=KS --save-dir=save_resnet164_c100 |& tee -a log_resnet164_c100_KS

# evaluate 
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar_resnet164.py  --arch=resnet164_c100  --pruning-method=MW --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=3 python -u trainer_cifar_resnet164.py  --arch=resnet164_c100  --pruning-method=KS --evaluate --print-freq=1
# ---------------------------------------------------------------------------------------------------------------------------------

### ImageNet

# ResNet50
# pretrained model
CUDA_VISIBLE_DEVICES=6 python -u trainer_imagenet_resnet50.py --save-dir=save_resnet50_imagenet |& tee -a log_resnet50_imagenet
CUDA_VISIBLE_DEVICES=2 python -u trainer_miniimagenet_resnet50.py --save-dir=save_resnet50_miniimagenet |& tee -a log_resnet50_miniimagenet
CUDA_VISIBLE_DEVICES=6 python -u trainer_imagenet_resnet50.py --pretrained --evaluate --print-freq=100
CUDA_VISIBLE_DEVICES=0 python -u trainer_miniimagenet_resnet50.py --pretrained --evaluate --print-freq=100

# pruning method 
CUDA_VISIBLE_DEVICES=3 python -u trainer_imagenet_resnet50.py --pruning-method=MW --save-dir=save_resnet50_imagenet |& tee -a log_resnet50_imagenet_MW
CUDA_VISIBLE_DEVICES=5 python -u trainer_imagenet_resnet50.py --pruning-method=EMD0.005 --save-dir=save_resnet50_imagenet |& tee -a log_resnet50_imagenet_emd0.005
CUDA_VISIBLE_DEVICES=1 python -u trainer_imagenet_resnet50.py --pruning-method=EMD0.01 --save-dir=save_resnet50_imagenet |& tee -a log_resnet50_imagenet_emd0.01
CUDA_VISIBLE_DEVICES=1 python -u trainer_imagenet_resnet50.py --pruning-method=EMD0.02 --save-dir=save_resnet50_imagenet |& tee -a log_resnet50_imagenet_emd0.02
CUDA_VISIBLE_DEVICES=1 python -u trainer_imagenet_resnet50.py --pruning-method=EMD0.05 --save-dir=save_resnet50_imagenet |& tee -a log_resnet50_imagenet_emd0.05
# CUDA_VISIBLE_DEVICES=5 python -u trainer_imagenet_resnet50.py --pruning-method=EMD0.1 --save-dir=save_resnet50_imagenet |& tee -a log_resnet50_imagenet_emd0.1
# CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_resnet50.py --pruning-method=RedCircle --save-dir=save_resnet50_imagenet |& tee -a log_resnet50_imagenet_RedCircle
CUDA_VISIBLE_DEVICES=6 python -u trainer_imagenet_resnet50.py --pruning-method=SRinit --save-dir=save_resnet50_imagenet |& tee -a log_resnet50_imagenet_SRinit
CUDA_VISIBLE_DEVICES=0 python -u trainer_imagenet_resnet50.py --pruning-method=MW --save-dir=save_resnet50_imagenet_RETRAIN |& tee -a log_resnet50_imagenet_MW_RETRAIN
CUDA_VISIBLE_DEVICES=1 python -u trainer_imagenet_resnet50.py --pruning-method=MW --save-dir=save_resnet50_imagenet_planA |& tee -a log_resnet50_imagenet_MW_planA
CUDA_VISIBLE_DEVICES=2 python -u trainer_imagenet_resnet50.py --pruning-method=MW --save-dir=save_resnet50_imagenet_planB |& tee -a log_resnet50_imagenet_MW_planB


# evaluate 
CUDA_VISIBLE_DEVICES=6 python -u trainer_imagenet_resnet50.py  --pruning-method=MW --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=5 python -u trainer_imagenet_resnet50.py  --pruning-method=EMD0.005 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=5 python -u trainer_imagenet_resnet50.py  --pruning-method=EMD0.01 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=5 python -u trainer_imagenet_resnet50.py  --pruning-method=EMD0.02 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_resnet50.py  --pruning-method=EMD0.05 --evaluate --print-freq=1
CUDA_VISIBLE_DEVICES=5 python -u trainer_imagenet_resnet50.py  --pruning-method=SRinit --evaluate --print-freq=1

# ResNet34
# pretrained model
CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_resnet34.py --pretrained --evaluate --print-freq=100
# pruning method 
CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_resnet34.py --pruning-method=MW --save-dir=save_resnet34_imagenet |& tee -a log_resnet34_imagenet_MW
# evaluate 
CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_resnet34.py --pruning-method=MW --evaluate --print-freq=100

# ResNet101
# pretrained model
CUDA_VISIBLE_DEVICES=6 python -u trainer_imagenet_resnet101.py --pretrained --evaluate --print-freq=100
# pruning method 
CUDA_VISIBLE_DEVICES=0 python -u trainer_imagenet_resnet101.py --pruning-method=MW --save-dir=save_resnet101_imagenet_planA |& tee -a log_resnet101_imagenet_MW_planA
CUDA_VISIBLE_DEVICES=1 python -u trainer_imagenet_resnet101.py --pruning-method=MW --save-dir=save_resnet101_imagenet_planB |& tee -a log_resnet101_imagenet_MW_planB
CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_resnet101.py --pruning-method=MW --save-dir=save_resnet101_imagenet |& tee -a log_resnet101_imagenet_MW

# evaluate 
CUDA_VISIBLE_DEVICES=6 python -u trainer_imagenet_resnet101.py --pruning-method=MW --evaluate --print-freq=100

# ResNet152
# pretrained model
CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_resnet152.py --pretrained --evaluate --print-freq=100
# pruning method 
CUDA_VISIBLE_DEVICES=2 python -u trainer_imagenet_resnet152.py --pruning-method=MW --save-dir=save_resnet152_imagenet |& tee -a log_resnet152_imagenet_MW
CUDA_VISIBLE_DEVICES=0 python -u trainer_imagenet_resnet152.py --pruning-method=KS --save-dir=save_resnet152_imagenet |& tee -a log_resnet152_imagenet_KS
# evaluate 
CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_resnet152.py --pruning-method=MW --evaluate --print-freq=100
CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_resnet152.py --pruning-method=KS --evaluate --print-freq=100

# MobileNetV2
# pretrained model
CUDA_VISIBLE_DEVICES=2 python -u trainer_imagenet_mobilenetv2.py --pretrained --evaluate --print-freq=100
# pruning method 
CUDA_VISIBLE_DEVICES=2 python -u trainer_imagenet_mobilenetv2.py --pruning-method=MW --save-dir=save_mobilenetv2_imagenet |& tee -a log_mobilenetv2_imagenet_MW
CUDA_VISIBLE_DEVICES=3 python -u trainer_imagenet_mobilenetv2.py --pruning-method=KS --save-dir=save_mobilenetv2_imagenet |& tee -a log_mobilenetv2_imagenet_KS
CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_mobilenetv2.py --save-dir=save_mobilenetv2_imagenet |& tee -a log_mobilenetv2_imagenet
# evaluate 
CUDA_VISIBLE_DEVICES=0 python -u trainer_imagenet_mobilenetv2.py --pruning-method=MW --evaluate --print-freq=100
CUDA_VISIBLE_DEVICES=5 python -u trainer_imagenet_mobilenetv2.py --pruning-method=KS --evaluate --print-freq=100

# ConvNeXt-T
# pretrained model
CUDA_VISIBLE_DEVICES=0 python -u trainer_imagenet_convnextt.py --pretrained --evaluate --print-freq=100
# pruning method 
CUDA_VISIBLE_DEVICES=0 python -u trainer_imagenet_convnextt.py --pruning-method=MW --save-dir=save_convnextt_imagenet |& tee -a log_convnextt_imagenet_MW
CUDA_VISIBLE_DEVICES=2 python -u trainer_imagenet_convnextt.py --pruning-method=KS --save-dir=save_convnextt_imagenet |& tee -a log_convnextt_imagenet_KS
# evaluate 
CUDA_VISIBLE_DEVICES=0 python -u trainer_imagenet_convnextt.py --pruning-method=MW --evaluate --print-freq=100
CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_convnextt.py --pruning-method=KS --evaluate --print-freq=100

# WideResNet50_2
# pretrained model
CUDA_VISIBLE_DEVICES=3 python -u trainer_imagenet_wideresnet50.py --pretrained --evaluate --print-freq=100
# pruning method 
CUDA_VISIBLE_DEVICES=0 python -u trainer_imagenet_wideresnet50.py --pruning-method=MW --save-dir=save_wideresnet50_imagenet |& tee -a log_wideresnet50_imagenet_MW
# evaluate 
CUDA_VISIBLE_DEVICES=4 python -u trainer_imagenet_wideresnet50.py --pruning-method=MW --evaluate --print-freq=100

# RegNet_Y
# pretrained model
CUDA_VISIBLE_DEVICES=2 python -u trainer_imagenet_regnetY.py --pretrained --evaluate --print-freq=100
# pruning method 
CUDA_VISIBLE_DEVICES=2 python -u trainer_imagenet_regnetY.py --pruning-method=MW --save-dir=save_regnetY_imagenet |& tee -a log_regnetY_imagenet_MW
# evaluate 
CUDA_VISIBLE_DEVICES=2 python -u trainer_imagenet_regnetY.py --pruning-method=MW --evaluate --print-freq=100

# EfficientNetV2S
# pretrained model
CUDA_VISIBLE_DEVICES=0 python -u trainer_imagenet_efficientnetv2s.py --pretrained --evaluate --print-freq=100
# pruning method 
CUDA_VISIBLE_DEVICES=2 python -u trainer_imagenet_efficientnetv2s.py --pruning-method=MW --save-dir=save_efficientnetv2s_imagenet |& tee -a log_efficientnetv2s_imagenet_MW
CUDA_VISIBLE_DEVICES=1 python -u trainer_imagenet_efficientnetv2s.py --pruning-method=KS --save-dir=save_efficientnetv2s_imagenet |& tee -a log_efficientnetv2s_imagenet_KS
# evaluate 
CUDA_VISIBLE_DEVICES=2 python -u trainer_imagenet_efficientnetv2s.py --pruning-method=MW --evaluate --print-freq=100
CUDA_VISIBLE_DEVICES=5 python -u trainer_imagenet_efficientnetv2s.py --pruning-method=KS --evaluate --print-freq=100


### TinyImageNet

# ResNet18
# pretrained model
CUDA_VISIBLE_DEVICES=6 python -u trainer_tinyimagenet_resnet18.py --pretrained --evaluate --print-freq=100
CUDA_VISIBLE_DEVICES=6 python -u trainer_tinyimagenet_resnet18.py --save-dir=save_resnet18_tinyimagenet |& tee -a log_resnet18_tinyimagenet

# ResNet34
# train
CUDA_VISIBLE_DEVICES=2 python -u trainer_tinyimagenet_resnet34.py --save-dir=save_resnet34_tinyimagenet |& tee -a log_resnet34_tinyimagenet
CUDA_VISIBLE_DEVICES=2 python -u trainer_tinyimagenet_resnet34.py --pruning-method=MW --save-dir=save_resnet34_tinyimagenet |& tee -a log_resnet34_tinyimagenet_MW
# evaluate 
CUDA_VISIBLE_DEVICES=6 python -u trainer_tinyimagenet_resnet34.py --pretrained --evaluate --print-freq=100
CUDA_VISIBLE_DEVICES=4 python -u trainer_tinyimagenet_resnet34.py --pruning-method=MW --evaluate --print-freq=100

# ResNet50
# train
CUDA_VISIBLE_DEVICES=2 python -u trainer_tinyimagenet_resnet50.py --save-dir=save_resnet50_tinyimagenet |& tee -a log_resnet50_tinyimagenet
CUDA_VISIBLE_DEVICES=0 python -u trainer_tinyimagenet_resnet50.py --pruning-method=MW --save-dir=save_resnet50_tinyimagenet |& tee -a log_resnet50_tinyimagenet_MW
# evaluate 
CUDA_VISIBLE_DEVICES=6 python -u trainer_tinyimagenet_resnet50.py --pretrained --evaluate --print-freq=100
CUDA_VISIBLE_DEVICES=1 python -u trainer_tinyimagenet_resnet50.py --pruning-method=MW --evaluate --print-freq=100

# ResNet101
# train
CUDA_VISIBLE_DEVICES=1 python -u trainer_tinyimagenet_resnet101.py --save-dir=save_resnet101_tinyimagenet |& tee -a log_resnet101_tinyimagenet
CUDA_VISIBLE_DEVICES=4 python -u trainer_tinyimagenet_resnet101.py --pruning-method=MW --save-dir=save_resnet101_tinyimagenet |& tee -a log_resnet101_tinyimagenet_MW
# evaluate 
CUDA_VISIBLE_DEVICES=2 python -u trainer_tinyimagenet_resnet101.py --pretrained --evaluate --print-freq=100
CUDA_VISIBLE_DEVICES=4 python -u trainer_tinyimagenet_resnet101.py --pruning-method=MW --evaluate --print-freq=100

# ResNet152
# train
CUDA_VISIBLE_DEVICES=1 python -u trainer_tinyimagenet_resnet152.py --save-dir=save_resnet152_tinyimagenet |& tee -a log_resnet152_tinyimagenet
CUDA_VISIBLE_DEVICES=5 python -u trainer_tinyimagenet_resnet152.py --pruning-method=MW --save-dir=save_resnet152_tinyimagenet |& tee -a log_resnet152_tinyimagenet_MW
# evaluate 
CUDA_VISIBLE_DEVICES=1 python -u trainer_tinyimagenet_resnet152.py --pretrained --evaluate --print-freq=100
CUDA_VISIBLE_DEVICES=5 python -u trainer_tinyimagenet_resnet152.py --pruning-method=MW --evaluate --print-freq=100