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
    import torch.quantization as quantization
    from torch.quantization import quantize_dynamic
    from datetime import datetime
    import time
    from pytorch_lightning.loggers import WandbLogger
    import pytorch_lightning as L

    # 初始化模型
    net = model_efficient.EFB4DFR.load_from_checkpoint(cfg["test"]["weights_path"])  # 加载权重字典

    # 設定設備
    device = torch.device("cpu")
    net.to(device)

    # 量化模型包裝類
    class QuantizedEfficientNetV2(nn.Module):
        def __init__(self, model):
            super(QuantizedEfficientNetV2, self).__init__()
            self.quant = quantization.QuantStub()
            self.model = model
            self.dequant = quantization.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.model(x)
            x = self.dequant(x)
            return x

    # 創建 EfficientNetV2 模型
    model = net.base_model.to(device)
    model.eval()

    # 設定量化後端
    torch.backends.quantized.engine = "onednn"
    print("目前使用的 Quantized Engine:", torch.backends.quantized.engine)

    # 設定 QConfig
    qconfig = quantization.get_default_qconfig("onednn")
    model.qconfig = qconfig  # 套用 per_tensor 設定

    # 讓 Depthwise Conv2d 保持 FP32
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.groups > 1:
            print(f"🚨 Depthwise Conv2d {name} groups={module.groups}，保持 FP32")
            module.qconfig = None

    # 讓模型進入量化準備模式
    model = quantization.prepare(model, inplace=True)
    model.to(device)

    # 用代表性數據進行校準 (Calibration)
    example_input = torch.randn(1, 3, 224, 224).to(device)
    model(example_input)

    # 轉換為真正的量化模型
    quantized_model = quantization.convert(model)
    quantized_model.to(device)

    # 儲存量化模型
    torch.save(quantized_model.state_dict(), "efficientnetv2_s_quantized.pth")
    print("Quantized model successfully created!")

    # 替換模型的 base_model
    net.base_model = QuantizedEfficientNetV2(quantized_model)
    net.to(device)

    # 測試前向傳播
    example_input = torch.randn(1, 3, 224, 224).to(device)
    output = net(example_input)
    if output is None:
        raise RuntimeError("🚨 forward 函數回傳 None，請檢查模型轉換流程！")
    else:
        print("✅ forward 正常運行，輸出形狀:", output.shape)

    # 設定 PyTorch Lightning 訓練器
    net.eval()
    date = datetime.now().strftime("%Y%m%d_%H%M")
    project = "paper2025"
    run_label = args.cfg.split("/")[-1].split(".")[0]
    run = cfg["dataset"]["name"] + f"_test_{date}_{run_label}"
    logger = WandbLogger(project=project, name=run, id=run, log_model=False)
    trainer = L.Trainer(
        accelerator="gpu" if "cuda" in str(device) else "cpu",
        devices=1,
        precision="16-mixed" if cfg["test"]["mixed_precision"] else 32,
        limit_test_batches=cfg["test"]["limit_test_batches"],
        logger=logger,
    )

    # 測試
    test_start_time = time.time()
    trainer.test(model=net, dataloaders=test_loader)
    test_end_time = time.time()
    print(f"Testing time: {test_end_time - test_start_time:.2f} seconds")
