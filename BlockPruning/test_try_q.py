from pprint import pprint
import argparse
import gc
from os.path import join
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from cifake_dataset import CIFAKEDataset

from coco_fake_dataset import COCOFakeDataset
from dffd_dataset import DFFDDataset
from ffpp_dataset import FaceForensicsDataset

# import model
import model_efficient
from lib.util import load_config
import random
import numpy as np
import os

import time

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        help="The path to the config.",
        default="./configs/ablation_baseline.cfg",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    args = args_func()

    # load configs
    cfg = load_config(args.cfg)
    pprint(cfg)

    # preliminary setup
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["model"]["gpu"])
    torch.manual_seed(cfg["test"]["seed"])
    random.seed(cfg["test"]["seed"])
    np.random.seed(cfg["test"]["seed"])
    torch.set_float32_matmul_precision("medium")

    # get data
    if cfg["dataset"]["name"] == "coco_fake":
        print(
            f"Load COCO-Fake datasets from {cfg['dataset']['coco2014_path']} and {cfg['dataset']['coco_fake_path']}"
        )
        test_dataset = COCOFakeDataset(
            coco2014_path=cfg["dataset"]["coco2014_path"],
            coco_fake_path=cfg["dataset"]["coco_fake_path"],
            split="val",
            mode="single",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "dffd":
        print(f"Load DFFD dataset from {cfg['dataset']['dffd_path']}")
        test_dataset = DFFDDataset(
            dataset_path=cfg["dataset"]["dffd_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "cifake":
        print(f"Loading CIFAKE dataset from {cfg['dataset']['cifake_path']}")
        test_dataset = CIFAKEDataset(
            dataset_path=cfg["dataset"]["cifake_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )
    elif cfg["dataset"]["name"] == "ff++c23":
        print(f"Loading FF++c23 dataset from {cfg['dataset']['ff++c23_path']}")
        test_dataset = FaceForensicsDataset(
            dataset_path=cfg["dataset"]["ff++c23_path"],
            split="test",
            resolution=cfg["test"]["resolution"],
        )

    # loads the dataloaders
    num_workers = 4
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["test"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    import torch
    import torch.nn as nn
    import torch.ao.quantization as quantization
    import torchvision.models as models
    import time
    from tqdm import tqdm

    # 確定設備
    device = torch.device("cpu")

    # 1. 加載 EfficientNetV2-S 預訓練模型
    model_fp32 = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).to(device)
    model_fp32.eval()

    # 2. 設定量化配置 (qconfig)
    model_fp32.qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.default_observer,
        weight=torch.ao.quantization.default_weight_observer
    )
    quantization.prepare(model_fp32, inplace=True)  # 進行準備 (收集統計數據)

    # 3. 執行 PTQ 校準 (用 test_loader 遍歷部分數據)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Calibrating Model", unit="batch"):
            images = batch["image"]
            model_fp32(images.to(device))
            break  # 只執行一個 batch 校準

    # 4. 轉換為量化模型
    model_quantized = quantization.convert(model_fp32)
    model_quantized.eval()

    # 測試函數
    def evaluate_model(model, dataloader, device):
        model.to(device)
        correct = 0
        total = 0
        start_time = time.time()

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"]
                labels = batch["is_real"]
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        elapsed_time = time.time() - start_time
        return accuracy, elapsed_time

    # 5. 進行推論測試 (未量化)
    accuracy_fp32, time_fp32 = evaluate_model(model_fp32, test_loader, device)

    # 6. 進行推論測試 (量化後)
    accuracy_quantized, time_quantized = evaluate_model(model_quantized, test_loader, device)

    # 返回結果
    {
        "FP32 Accuracy": accuracy_fp32,
        "FP32 Inference Time (s)": time_fp32,
        "Quantized Accuracy": accuracy_quantized,
        "Quantized Inference Time (s)": time_quantized
    }
