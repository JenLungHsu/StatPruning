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
from celeb_dataset import CelebDFDataset

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
    elif cfg["dataset"]["name"] == "celebdf":
        print(f"Loading CelebDF dataset from {cfg['dataset']['celebdf_path']}")
        test_dataset = CelebDFDataset(
            dataset_path=cfg["dataset"]["celebdf_path"],
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

    # init model
    # net = model_efficient.EFB4DFR.load_from_checkpoint(join(cfg["test"]["weights_path"], f"{cfg['dataset']['name']}_{cfg['model']['backbone'][-1]}{'_unfrozen' if not cfg['model']['freeze_backbone'] else ''}.ckpt"))
    net = model_efficient.EFB4DFR.load_from_checkpoint(cfg["test"]["weights_path"])  # 加載權重字典

    # ----
    import time
    import torch
    from lightning.pytorch.callbacks import Callback

    class LatencyCallback(Callback):
        def __init__(self):
            super().__init__()
            self.batch_times = []
            self.use_event = torch.cuda.is_available()
            if self.use_event:
                self.starter = torch.cuda.Event(enable_timing=True)
                self.ender   = torch.cuda.Event(enable_timing=True)

        # 這裡把 dataloader_idx 變成可選，或直接用 *args 接
        def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
            if self.use_event:
                self.starter.record()
            else:
                self._t0 = time.time()

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
            if self.use_event:
                self.ender.record()
                torch.cuda.synchronize()
                self.batch_times.append(self.starter.elapsed_time(self.ender) / 1000.0)
            else:
                self.batch_times.append(time.time() - self._t0)

        def on_test_end(self, trainer, pl_module):
            avg = sum(self.batch_times) / len(self.batch_times)
            print(f"\n>>> 平均每 batch 推論時間：{avg*1000:.2f} ms （{1.0/avg:.1f} batch/s）")

    # ----

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # start training
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
        callbacks=[LatencyCallback()],
    )

    test_start_time = time.time()
    trainer.test(model=net, dataloaders=test_loader)
    test_end_time = time.time()
    test_time = test_end_time - test_start_time
    print(f"Testing time: {test_time:.2f} seconds")

    # # ------ 在正式測量前做 warm-up ------
    # warmup_batches = 5
    # net.eval()
    # with torch.no_grad():
    #     for i, batch in enumerate(test_loader):
    #         if i >= warmup_batches:
    #             break
    #         _ = net(batch['image'].to(device))
    #         if device.type == 'cuda':
    #             torch.cuda.synchronize()

    # # ------ 開始正式測量 batch latency ------
    # batch_times = []
    # net.eval()
    # with torch.no_grad():
    #     for batch in test_loader:
    #         inputs = batch['image'].to(device)

    #         if device.type == 'cuda':
    #             # GPU 時用 CUDA Event
    #             starter = torch.cuda.Event(enable_timing=True)
    #             ender   = torch.cuda.Event(enable_timing=True)
    #             starter.record()
    #             _ = net(inputs)
    #             ender.record()
    #             torch.cuda.synchronize()
    #             # elapsed_time 回傳毫秒
    #             batch_times.append(starter.elapsed_time(ender) / 1000.0)  # 轉成秒
    #         # else:
    #         #     # CPU 時用 time.time
    #         #     start = time.time()
    #         #     _ = net(inputs)
    #         #     elapsed = time.time() - start
    #         #     batch_times.append(elapsed)

    # # ------ 計算並印出平均 batch latency ------
    # avg_batch_time = sum(batch_times) / len(batch_times)
    # print(f"Average inference time per batch: {avg_batch_time:.5f} s  "
    #     f"({1.0/avg_batch_time:.1f} batches/sec)")
