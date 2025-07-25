import gc
import cv2 as cv
import numpy as np 
import einops
from skimage import feature

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchmetrics.functional.classification import accuracy, auroc
import timm
import lightning as L
from fvcore.nn import FlopCountAnalysis, parameter_count

# from BNext.src.bnext import BNext
from pytorchcv.model_provider import get_model as ptcv_get_model 

# import sys
# sys.path.append("/ssd5/Roy/BlockPruning")
from ModifiedEFB4 import ModifiedEFB4
from ModifiedEFB0_EMD2 import ModifiedEFB0_EMD2
from ModifiedEFB0_EMD3 import ModifiedEFB0_EMD3
from ModifiedEFB0_EMD4 import ModifiedEFB0_EMD4

from EffV2_S.EffV2_S_MW import EffV2_S_MW
from EffV2_S.EffV2_S_KS import EffV2_S_KS
from EffV2_S.EffV2_S_EMD02 import EffV2_S_EMD02
from EffV2_S.EffV2_S_EMD03 import EffV2_S_EMD03
from EffV2_S.EffV2_S_EMD04 import EffV2_S_EMD04
from EffV2_S.EffV2_S_EMD10 import EffV2_S_EMD10
from EffV2_S.EffV2_S_RedCircle import EffV2_S_RedCircle

from EffV2_M.EffV2_M_MW import EffV2_M_MW
from EffV2_M.EffV2_M_KS import EffV2_M_KS
from EffV2_M.EffV2_M_EMD02 import EffV2_M_EMD02
from EffV2_M.EffV2_M_EMD03 import EffV2_M_EMD03
from EffV2_M.EffV2_M_EMD04 import EffV2_M_EMD04
from EffV2_M.EffV2_M_EMD10 import EffV2_M_EMD10
from EffV2_M.EffV2_M_RedCircle import EffV2_M_RedCircle

from EffV2_L.EffV2_L_MW import EffV2_L_MW
from EffV2_L.EffV2_L_KS import EffV2_L_KS
from EffV2_L.EffV2_L_EMD02 import EffV2_L_EMD02
from EffV2_L.EffV2_L_EMD03 import EffV2_L_EMD03
from EffV2_L.EffV2_L_EMD04 import EffV2_L_EMD04
from EffV2_L.EffV2_L_EMD10 import EffV2_L_EMD10
from EffV2_L.EffV2_L_RedCircle import EffV2_L_RedCircle

from EffV2_XL.EffV2_XL_MW import EffV2_XL_MW
from EffV2_XL.EffV2_XL_KS import EffV2_XL_KS
from EffV2_XL.EffV2_XL_EMD02 import EffV2_XL_EMD02
from EffV2_XL.EffV2_XL_EMD03 import EffV2_XL_EMD03
from EffV2_XL.EffV2_XL_EMD04 import EffV2_XL_EMD04
from EffV2_XL.EffV2_XL_EMD10 import EffV2_XL_EMD10
from EffV2_XL.EffV2_XL_RedCircle import EffV2_XL_RedCircle

# import sys
# sys.path.append("/ssd5/Roy/BlockPruning/ResNet")
# from small_resnet import resnet56
# ---------------------------------------------------------------------------

class EFB4DFR(L.LightningModule):

    def __init__(self, 
                 num_classes, 
                 backbone='EFB4', 
                 freeze_backbone=False, 
                 add_magnitude_channel=False,
                 add_phase_channel=False,
                 learning_rate=1e-4, 
                 pos_weight=1.):
        super(EFB4DFR, self).__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epoch_outs = []
         
        # loads the backbone
        self.backbone = backbone
        if backbone == "EFB4":
            print(backbone)
            self.base_model = ptcv_get_model("efficientnet_b4c", pretrained=True)
            # finetuned_weights_path = "/ssd6/Roy/Few-Shot-Forgery-Detection/ckpt/2_EfficientNetB4_pytorch.pkl"
            # checkpoint = torch.load(finetuned_weights_path)  # 使用 torch.load 讀取檔案
            # self.base_model.load_state_dict(checkpoint, strict=False)  # 加載權重字典
                
        elif backbone == "EFB0":
            print(backbone)
            self.base_model = ptcv_get_model("efficientnet_b0", pretrained=True)
            # finetuned_weights_path = "/ssd6/Roy/Few-Shot-Forgery-Detection/ckpt/2_EfficientNetB4_pytorch.pkl"
            # checkpoint = torch.load(finetuned_weights_path)  # 使用 torch.load 讀取檔案
            # self.base_model.load_state_dict(checkpoint, strict=False)  # 加載權重字典

        elif backbone == "EFB4_pruned":
            print(backbone)
            original_model = ptcv_get_model("efficientnet_b4c", pretrained=True)
            # 載入你的 finetuned 權重
            # finetuned_weights_path = "/ssd6/Roy/Few-Shot-Forgery-Detection/ckpt/EfficientNetB4_100%_6epoch.pkl"
            # checkpoint = torch.load(finetuned_weights_path)
            # original_model.load_state_dict(checkpoint, strict=False)

            # 建立新模型，只保留你想要的部分
            self.base_model = ModifiedEFB4(original_model)

        elif backbone == "EffV2_s":
            print(backbone)
            import torchvision.models as models
            # 如果想要使用預訓練權重，可以傳入 weights 參數
            self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # elif backbone == "resnet56_cifar10":
        #     print(backbone)
        #     self.base_model = resnet56(num_classes=10)
        #     checkpoint = torch.load("/ssd5/Roy/BlockPruning/pretrained/resnet56-4bfd9763.th", map_location="cpu")
        #     state_dict = checkpoint["state_dict"]
        #     new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        #     self.base_model.load_state_dict(new_state_dict)
        #     print(self.base_model)

        elif backbone.startswith("EffV2_"):
            print(backbone)

            import timm
            model_map = {
                "EffV2_S": "tf_efficientnetv2_s_in21k",
                "EffV2_M": "tf_efficientnetv2_m_in21k",
                "EffV2_L": "tf_efficientnetv2_l_in21k",
                "EffV2_XL": "tf_efficientnetv2_xl_in21k",
            }

            if backbone in model_map:
                self.base_model = timm.create_model(model_map[backbone], pretrained=True)
            else:
                function_mapping = {"EffV2_S_MW": EffV2_S_MW,
                                    "EffV2_S_KS": EffV2_S_KS,
                                    "EffV2_S_EMD02": EffV2_S_EMD02,
                                    "EffV2_S_EMD03": EffV2_S_EMD03,
                                    "EffV2_S_EMD04": EffV2_S_EMD04,
                                    "EffV2_S_EMD10": EffV2_S_EMD10,
                                    "EffV2_S_RedCircle": EffV2_S_RedCircle,

                                    "EffV2_M_MW": EffV2_M_MW,
                                    "EffV2_M_KS": EffV2_M_KS,
                                    "EffV2_M_EMD02": EffV2_M_EMD02,
                                    "EffV2_M_EMD03": EffV2_M_EMD03,
                                    "EffV2_M_EMD04": EffV2_M_EMD04,
                                    "EffV2_M_EMD10": EffV2_M_EMD10,
                                    "EffV2_M_RedCircle": EffV2_M_RedCircle,

                                    "EffV2_L_MW": EffV2_L_MW,
                                    "EffV2_L_KS": EffV2_L_KS,
                                    "EffV2_L_EMD02": EffV2_L_EMD02,
                                    "EffV2_L_EMD03": EffV2_L_EMD03,
                                    "EffV2_L_EMD04": EffV2_L_EMD04,
                                    "EffV2_L_EMD10": EffV2_L_EMD10,
                                    "EffV2_L_RedCircle": EffV2_L_RedCircle,

                                    "EffV2_XL_MW": EffV2_XL_MW,
                                    "EffV2_XL_KS": EffV2_XL_KS,
                                    "EffV2_XL_EMD02": EffV2_XL_EMD02,
                                    "EffV2_XL_EMD03": EffV2_XL_EMD03,
                                    "EffV2_XL_EMD04": EffV2_XL_EMD04,
                                    "EffV2_XL_EMD10": EffV2_XL_EMD10,
                                    "EffV2_XL_RedCircle": EffV2_XL_RedCircle,
                                }
                self.base_model = function_mapping[backbone]()  # 呼叫對應函式

        # elif backbone.startswith("EffV2_"):
        #     print(backbone)
        #     import timm
        #     model_map = {
        #         "EffV2_S": "tf_efficientnetv2_s_in21k",
        #         "EffV2_M": "tf_efficientnetv2_m_in21k",
        #         "EffV2_L": "tf_efficientnetv2_l_in21k",
        #         "EffV2_XL": "tf_efficientnetv2_xl_in21k",
        #     }

        #     if backbone in model_map:
        #         self.base_model = timm.create_model(model_map[backbone], pretrained=True)
        #     else:
        #         raise ValueError(f"未知的 backbone: {backbone}")
            
        # elif backbone == "EffV2_s_KS_0.05":
        #     print(backbone)
        #     import torchvision.models as models
        #     # 如果想要使用預訓練權重，可以傳入 weights 參數
        #     self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        #     # finetuned_weights_path = "/ssd5/Roy/binary_deepfake_detection/paper2025/ff++c23_20250302_1538_results_ff++c23_EffV2_s_test_auc_5epoch/checkpoints/ff++c23_EffV2_s_epoch=1-train_acc=1.00-val_acc=0.94.ckpt"
        #     # checkpoint = torch.load(finetuned_weights_path, weights_only=True)
        #     # self.base_model.load_state_dict(checkpoint, strict=False)

        #     for i in range(14, 9, -1):  # 逆向刪除，避免索引錯亂
        #         del self.base_model.features[6][i]
        #     del self.base_model.features[6][4]  # 刪除 features[6][4]
        #     del self.base_model.features[5][8]  # 刪除 features[5][8]
        #     del self.base_model.features[5][1]  # 刪除 features[5][1]

        # elif backbone == "EffV2_s_EMD_0.2":
        #     print(backbone)
        #     import torchvision.models as models
        #     # 如果想要使用預訓練權重，可以傳入 weights 參數
        #     self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        #     # finetuned_weights_path = "/ssd5/Roy/binary_deepfake_detection/paper2025/ff++c23_20250302_1538_results_ff++c23_EffV2_s_test_auc_5epoch/checkpoints/ff++c23_EffV2_s_epoch=1-train_acc=1.00-val_acc=0.94.ckpt"
        #     # checkpoint = torch.load(finetuned_weights_path, weights_only=True)
        #     # self.base_model.load_state_dict(checkpoint, strict=False)

        #     for i in range(14, 5, -1):  # 逆向刪除，避免索引錯亂
        #         del self.base_model.features[6][i]
        #     del self.base_model.features[6][4]  # 刪除 features[6][4]
        #     del self.base_model.features[5][8]  # 刪除 features[5][8]
        #     del self.base_model.features[5][7]  # 刪除 features[5][7]
        #     del self.base_model.features[5][6]  # 刪除 features[5][6]
        #     del self.base_model.features[5][1]  # 刪除 features[5][1]
        #     del self.base_model.features[4][5]  # 刪除 features[4][5]
        #     del self.base_model.features[4][4]  # 刪除 features[4][4]

        # elif backbone == "EffV2_s_EMD_0.25":
        #     print(backbone)
        #     import torchvision.models as models
        #     # 如果想要使用預訓練權重，可以傳入 weights 參數
        #     self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        #     # finetuned_weights_path = "/ssd5/Roy/binary_deepfake_detection/paper2025/ff++c23_20250302_1538_results_ff++c23_EffV2_s_test_auc_5epoch/checkpoints/ff++c23_EffV2_s_epoch=1-train_acc=1.00-val_acc=0.94.ckpt"
        #     # checkpoint = torch.load(finetuned_weights_path, weights_only=True)
        #     # self.base_model.load_state_dict(checkpoint, strict=False)

        #     for i in range(14, 2, -1):  # 逆向刪除，避免索引錯亂
        #         del self.base_model.features[6][i]

        #     del self.base_model.features[5][8]  # 刪除 features[5][8]
        #     del self.base_model.features[5][7]  # 刪除 features[5][7]
        #     del self.base_model.features[5][6]  # 刪除 features[5][6]
        #     del self.base_model.features[5][5]  # 刪除 features[5][5]
        #     del self.base_model.features[5][4]  # 刪除 features[5][4]
        #     del self.base_model.features[5][1]  # 刪除 features[5][1]

        #     del self.base_model.features[4][5]  # 刪除 features[4][5]
        #     del self.base_model.features[4][4]  # 刪除 features[4][4]
        #     del self.base_model.features[4][3]  # 刪除 features[4][3]

        #     del self.base_model.features[3][3]  # 刪除 features[3][3]
        #     del self.base_model.features[2][3]  # 刪除 features[2][3]
        #     del self.base_model.features[1][1]  # 刪除 features[1][1]

        # elif backbone == "EffV2S_EMD0.3":
        #     print(backbone)
        #     import torchvision.models as models
        #     # 如果想要使用預訓練權重，可以傳入 weights 參數
        #     self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        #     # finetuned_weights_path = "/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250302_1538_results_ff++c23_EffV2_s_test_auc_5epoch/checkpoints/ff++c23_EffV2_s_epoch=1-train_acc=1.00-val_acc=0.94.ckpt"
        #     # checkpoint = torch.load(finetuned_weights_path, weights_only=True)
        #     # self.base_model.load_state_dict(checkpoint, strict=False)

        #     for i in range(14, 1, -1):  # 逆向刪除，避免索引錯亂
        #         del self.base_model.features[6][i]

        #     del self.base_model.features[5][8]  # 刪除 features[5][8]
        #     del self.base_model.features[5][7]  # 刪除 features[5][7]
        #     del self.base_model.features[5][6]  # 刪除 features[5][6]
        #     del self.base_model.features[5][5]  # 刪除 features[5][5]
        #     del self.base_model.features[5][4]  # 刪除 features[5][4]
        #     del self.base_model.features[5][3]  # 刪除 features[5][3]
        #     del self.base_model.features[5][1]  # 刪除 features[5][1]

        #     del self.base_model.features[4][5]  # 刪除 features[4][5]
        #     del self.base_model.features[4][4]  # 刪除 features[4][4]
        #     del self.base_model.features[4][3]  # 刪除 features[4][3]
        #     del self.base_model.features[4][2]  # 刪除 features[4][2]

        #     del self.base_model.features[3][3]  # 刪除 features[3][3]
        #     del self.base_model.features[3][2]  # 刪除 features[3][2]

        #     del self.base_model.features[2][3]  # 刪除 features[2][3]
        #     del self.base_model.features[2][2]  # 刪除 features[2][2]

        #     del self.base_model.features[1][1]  # 刪除 features[1][1]

        # elif backbone == "EffV2S_EMD0.35":
        #     print(backbone)
        #     import torchvision.models as models
        #     # 如果想要使用預訓練權重，可以傳入 weights 參數
        #     self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        #     # finetuned_weights_path = "/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250302_1538_results_ff++c23_EffV2_s_test_auc_5epoch/checkpoints/ff++c23_EffV2_s_epoch=1-train_acc=1.00-val_acc=0.94.ckpt"
        #     # checkpoint = torch.load(finetuned_weights_path, weights_only=True)
        #     # self.base_model.load_state_dict(checkpoint, strict=False)

        #     for i in range(14, 1, -1):  # 逆向刪除，避免索引錯亂
        #         del self.base_model.features[6][i]

        #     del self.base_model.features[5][8]  # 刪除 features[5][8]
        #     del self.base_model.features[5][7]  # 刪除 features[5][7]
        #     del self.base_model.features[5][6]  # 刪除 features[5][6]
        #     del self.base_model.features[5][5]  # 刪除 features[5][5]
        #     del self.base_model.features[5][4]  # 刪除 features[5][4]
        #     del self.base_model.features[5][3]  # 刪除 features[5][3]
        #     del self.base_model.features[5][2]  # 刪除 features[5][2]
        #     del self.base_model.features[5][1]  # 刪除 features[5][1]

        #     del self.base_model.features[4][5]  # 刪除 features[4][5]
        #     del self.base_model.features[4][4]  # 刪除 features[4][4]
        #     del self.base_model.features[4][3]  # 刪除 features[4][3]
        #     del self.base_model.features[4][2]  # 刪除 features[4][2]
        #     del self.base_model.features[4][1]  # 刪除 features[4][1]

        #     del self.base_model.features[3][3]  # 刪除 features[3][3]
        #     del self.base_model.features[3][2]  # 刪除 features[3][2]
        #     del self.base_model.features[3][1]  # 刪除 features[3][1]

        #     del self.base_model.features[2][3]  # 刪除 features[2][3]
        #     del self.base_model.features[2][2]  # 刪除 features[2][2]
        #     del self.base_model.features[2][1]  # 刪除 features[2][1]

        #     del self.base_model.features[1][1]  # 刪除 features[1][1]

        # elif backbone == "EffV2S_EMD0.4":
        #     print(backbone)
        #     import torchvision.models as models
        #     # 如果想要使用預訓練權重，可以傳入 weights 參數
        #     self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        #     # finetuned_weights_path = "/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250302_1538_results_ff++c23_EffV2_s_test_auc_5epoch/checkpoints/ff++c23_EffV2_s_epoch=1-train_acc=1.00-val_acc=0.94.ckpt"
        #     # checkpoint = torch.load(finetuned_weights_path, weights_only=True)
        #     # self.base_model.load_state_dict(checkpoint, strict=False)

        #     for i in range(14, 0, -1):  # 逆向刪除，避免索引錯亂
        #         del self.base_model.features[6][i]

        #     del self.base_model.features[5][8]  # 刪除 features[5][8]
        #     del self.base_model.features[5][7]  # 刪除 features[5][7]
        #     del self.base_model.features[5][6]  # 刪除 features[5][6]
        #     del self.base_model.features[5][5]  # 刪除 features[5][5]
        #     del self.base_model.features[5][4]  # 刪除 features[5][4]
        #     del self.base_model.features[5][3]  # 刪除 features[5][3]
        #     del self.base_model.features[5][2]  # 刪除 features[5][2]
        #     del self.base_model.features[5][1]  # 刪除 features[5][1]

        #     del self.base_model.features[4][5]  # 刪除 features[4][5]
        #     del self.base_model.features[4][4]  # 刪除 features[4][4]
        #     del self.base_model.features[4][3]  # 刪除 features[4][3]
        #     del self.base_model.features[4][2]  # 刪除 features[4][2]
        #     del self.base_model.features[4][1]  # 刪除 features[4][1]

        #     del self.base_model.features[3][3]  # 刪除 features[3][3]
        #     del self.base_model.features[3][2]  # 刪除 features[3][2]
        #     del self.base_model.features[3][1]  # 刪除 features[3][1]

        #     del self.base_model.features[2][3]  # 刪除 features[2][3]
        #     del self.base_model.features[2][2]  # 刪除 features[2][2]
        #     del self.base_model.features[2][1]  # 刪除 features[2][1]

        #     del self.base_model.features[1][1]  # 刪除 features[1][1]

        # elif backbone == "EffV2_s_Mann_0.05":
        #     print(backbone)
        #     import torchvision.models as models
        #     # 如果想要使用預訓練權重，可以傳入 weights 參數
        #     self.base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        #     # finetuned_weights_path = "/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250302_1538_results_ff++c23_EffV2_s_test_auc_5epoch/checkpoints/ff++c23_EffV2_s_epoch=1-train_acc=1.00-val_acc=0.94.ckpt"
        #     # checkpoint = torch.load(finetuned_weights_path, weights_only=True)
        #     # self.base_model.load_state_dict(checkpoint, strict=False)

        #     for i in range(14, -1, -1):  # 逆向刪除，避免索引錯亂
        #         del self.base_model.features[6][i]

        #     del self.base_model.features[5][8]  # 刪除 features[5][8]
        #     del self.base_model.features[5][7]  # 刪除 features[5][7]
        #     del self.base_model.features[5][6]  # 刪除 features[5][6]
        #     del self.base_model.features[5][5]  # 刪除 features[5][5]
        #     del self.base_model.features[5][4]  # 刪除 features[5][4]
        #     del self.base_model.features[5][3]  # 刪除 features[5][3]
        #     del self.base_model.features[5][2]  # 刪除 features[5][2]
        #     del self.base_model.features[5][1]  # 刪除 features[5][1]
        #     del self.base_model.features[5][0]  # 刪除 features[5][0]

        #     del self.base_model.features[4][5]  # 刪除 features[4][5]
        #     del self.base_model.features[4][4]  # 刪除 features[4][4]
        #     del self.base_model.features[4][3]  # 刪除 features[4][3]
        #     del self.base_model.features[4][2]  # 刪除 features[4][2]
        #     del self.base_model.features[4][1]  # 刪除 features[4][1]
        #     del self.base_model.features[4][0]  # 刪除 features[4][0]

        #     del self.base_model.features[3][3]  # 刪除 features[3][3]
        #     del self.base_model.features[3][2]  # 刪除 features[3][2]
        #     del self.base_model.features[3][1]  # 刪除 features[3][1]
        #     del self.base_model.features[3][0]  # 刪除 features[3][0]

        #     del self.base_model.features[2][3]  # 刪除 features[2][3]
        #     del self.base_model.features[2][2]  # 刪除 features[2][2]
        #     del self.base_model.features[2][1]  # 刪除 features[2][1]
            
        #     # 取得原本的 Conv2d 層
        #     old_conv = self.base_model.features[7][0]
        #     # 重新建立一個新的 Conv2d 層，修改 in_channels 為 48，其它參數保持不變
        #     new_conv = torch.nn.Conv2d(
        #         in_channels=48,  # 修改這裡
        #         out_channels=old_conv.out_channels,
        #         kernel_size=old_conv.kernel_size,
        #         stride=old_conv.stride,
        #         padding=old_conv.padding,
        #         bias=old_conv.bias is not None  # 保持 bias 設定
        #     )
        #     # 替換舊的 Conv2d 層
        #     self.base_model.features[7][0] = new_conv

        # elif backbone == "EffV2M":
        #     print(backbone)
        #     import torchvision.models as models
        #     # 如果想要使用預訓練權重，可以傳入 weights 參數
        #     self.base_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)

        # elif backbone == "EFB0_EMD_0.2":
        #     print(backbone)
        #     original_model = ptcv_get_model("efficientnet_b0", pretrained=True)
        #     # 載入你的 finetuned 權重
        #     finetuned_weights_path = "/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250219_1054_results_ff++c23_EFB0_test_auc_5epoch/checkpoints/ff++c23_EFB0_epoch=1-train_acc=0.99-val_acc=0.93.ckpt"
        #     checkpoint = torch.load(finetuned_weights_path)
        #     original_model.load_state_dict(checkpoint, strict=False)

        #     # 建立新模型，只保留你想要的部分
        #     self.base_model = ModifiedEFB0_EMD2(original_model)

        # elif backbone == "EFB0_EMD_0.3":
        #     print(backbone)
        #     original_model = ptcv_get_model("efficientnet_b0", pretrained=True)
        #     # 載入你的 finetuned 權重
        #     finetuned_weights_path = "/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250219_1054_results_ff++c23_EFB0_test_auc_5epoch/checkpoints/ff++c23_EFB0_epoch=1-train_acc=0.99-val_acc=0.93.ckpt"
        #     checkpoint = torch.load(finetuned_weights_path)
        #     original_model.load_state_dict(checkpoint, strict=False)

        #     # 建立新模型，只保留你想要的部分
        #     self.base_model = ModifiedEFB0_EMD3(original_model)

        # elif backbone == "EFB0_EMD_0.4":
        #     print(backbone)
        #     original_model = ptcv_get_model("efficientnet_b0", pretrained=True)
        #     # 載入你的 finetuned 權重
        #     finetuned_weights_path = "/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250219_1054_results_ff++c23_EFB0_test_auc_5epoch/checkpoints/ff++c23_EFB0_epoch=1-train_acc=0.99-val_acc=0.93.ckpt"
        #     checkpoint = torch.load(finetuned_weights_path)
        #     original_model.load_state_dict(checkpoint, strict=False)

        #     # 建立新模型，只保留你想要的部分
        #     self.base_model = ModifiedEFB0_EMD4(original_model)

        else:
            raise ValueError("Unsupported Backbone!")
    
        # update the preprocessing metas
        assert isinstance(add_magnitude_channel, bool)
        self.add_magnitude_channel = add_magnitude_channel
        assert isinstance(add_phase_channel, bool)
        self.add_phase_channel = add_phase_channel
        self.new_channels = sum([self.add_magnitude_channel, self.add_phase_channel])
        print("self.new_channels:",self.new_channels)
        
        # loss parameters
        self.pos_weight = pos_weight

        # 確保我們修改的是正確的層
        if self.new_channels:
            # EfficientNet 的初始層是包含卷積層的第一個區塊（EffiInitBlock）
            original_input_layer = self.base_model.features.init_block.conv.conv  # 訪問最初的卷積層

            # 使用新的通道數創建一個新的卷積層
            new_conv_layer = nn.Conv2d(
                3+self.new_channels,  # 設置新的輸入通道數
                original_input_layer.out_channels,  # 保持輸出通道不變
                kernel_size=original_input_layer.kernel_size,  # 保持原來的卷積核大小
                stride=original_input_layer.stride,  # 保持原來的步長
                padding=original_input_layer.padding  # 保持原來的填充方式
            )

            # 用新創建的卷積層替換原始的卷積層
            self.base_model.features.init_block.conv.conv = new_conv_layer
            
        if not backbone.startswith("EffV2"):
            # disables the last layer of the backbone
            self.inplanes = self.base_model.output.fc.in_features
            self.base_model.deactive_last_layer=True
            self.base_model.output.fc = nn.Identity()
            # print(self.base_model)
        else:
            if backbone in ["EffV2_S","EffV2_M","EffV2_L","EffV2_XL", "EffV2_S_DyT"]:
                self.inplanes = self.base_model.classifier.in_features
                self.base_model.deactive_last_layer=True
                self.base_model.classifier = nn.Identity()
                # print(self.base_model)
            else:
                self.inplanes = 1280
                # self.inplanes = self.base_model.classifier[1].in_features
                # self.base_model.deactive_last_layer=True
                # self.base_model.classifier[1] = nn.Identity()
                # print(self.base_model)

        # eventually freeze the backbone
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.base_model.parameters():
                p.requires_grad = False

        # add a new linear layer after the backbone
        self.fc = nn.Linear(self.inplanes, num_classes if num_classes >= 3 else 1)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        outs = {}
        # eventually concat the edge sharpness to the input image in the channel dimension
        if self.add_magnitude_channel or self.add_phase_channel:
            x = self.add_new_channels(x)
        
        # normalizes the input image
        if self.new_channels == 3:
            x = (x - torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_MEAN, device=self.device).view(1, -1, 1, 1)) / torch.as_tensor(timm.data.constants.IMAGENET_DEFAULT_STD, device=self.device).view(1, -1, 1, 1)

        features = self.base_model(x)
        
        # outputs the logits
        outs["logits"] = self.fc(features)
        return outs
    
    def configure_optimizers(self):
        modules_to_train = [self.fc]
        if not self.freeze_backbone:
            modules_to_train.append(self.base_model)
        optimizer = optim.AdamW(
            [parameter for module in modules_to_train for parameter in module.parameters()], 
            lr=self.learning_rate,
            )
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0.1, total_iters=5)
        return [optimizer], [scheduler]
    
    def _add_new_channels_worker(self, image):
        # convert the image to grayscale
        gray = cv.cvtColor((image.cpu().numpy() * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)

        img_magnitude, img_phase = self.magnitude_phase_spectrum_transform(image, use_magnitude=self.add_magnitude_channel, use_phase=self.add_phase_channel)

        new_channels = []
        if self.add_magnitude_channel:
            new_channels.append(img_magnitude)
        
        if self.add_phase_channel:
            new_channels.append(img_phase)

        new_channels = np.stack(new_channels, axis=2) / 255
        return torch.from_numpy(new_channels).to(self.device).float()
        
    def add_new_channels(self, images):
        #copy the input image to avoid modifying the originalu
        images_copied = einops.rearrange(images, "b c h w -> b h w c")
        
        # parallelize over each image in the batch using pool
        new_channels = torch.stack([self._add_new_channels_worker(image) for image in images_copied], dim=0)
        
        # concatenates the new channels to the input image in the channel dimension
        images_copied = torch.concatenate([images_copied, new_channels], dim=-1)
        # cast img again to torch tensor and then reshape to (B, C, H, W)
        images_copied = einops.rearrange(images_copied, "b h w c -> b c h w")
        return images_copied
    
    def magnitude_phase_spectrum_transform(self, image, use_magnitude=True, use_phase=True):
        """
        將圖像轉換到頻率域，然後選擇性地保留幅度譜或相位譜，再轉換回空間域。
        
        :param image: 輸入的圖像
        :param use_magnitude: 是否保留幅度譜
        :param use_phase: 是否保留相位譜
        :return: 轉換後的圖像，通道數根據選擇決定
        """
        # 將圖像轉為灰度圖
        gray = cv.cvtColor((image.cpu().numpy() * 255).astype(np.uint8), cv.COLOR_BGR2GRAY)
        
        # 對圖像進行傅立葉變換
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)  # 將零頻率成分移至中心

        # 提取幅度譜和相位譜
        magnitude = np.abs(fshift)  # 幅度譜
        phase = np.angle(fshift)  # 相位譜

        img_magnitude, img_phase = 0, 0

        # 如果需要保留幅度譜，則將其轉換回空間域
        if use_magnitude:
            magnitude_fshift = magnitude * np.exp(1j * np.zeros_like(phase))  # 保留幅度，設置相位為零
            f_ishift = np.fft.ifftshift(magnitude_fshift)  # 恢復頻率成分的原始位置
            img_magnitude = np.fft.ifft2(f_ishift)  # 執行逆傅立葉變換
            img_magnitude = np.abs(img_magnitude)  # 取絕對值

        # 如果需要保留相位譜，則將其轉換回空間域
        if use_phase:
            phase_fshift = np.exp(1j * phase)  # 保留相位，設置幅度為1
            f_ishift = np.fft.ifftshift(phase_fshift)  # 恢復頻率成分的原始位置
            img_phase = np.fft.ifft2(f_ishift)  # 執行逆傅立葉變換
            img_phase = np.abs(img_phase)  # 取絕對值
        
        return img_magnitude, img_phase
    
    def on_train_start(self):
        return self._on_start()
    
    def on_test_start(self):
        return self._on_start()
    
    def on_train_epoch_start(self):
        self._on_epoch_start()
        
    def on_test_epoch_start(self):
        self._on_epoch_start()
        
    def training_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="train")
    
    def validation_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="val")
    
    def test_step(self, batch, i_batch):
        return self._step(batch, i_batch, phase="test")
    
    def on_train_epoch_end(self):
        self._on_epoch_end()
        
    def on_test_epoch_end(self):
        self._on_epoch_end()
    
    def _step(self, batch, i_batch, phase=None):
        images = batch["image"].to(self.device)
        outs = {
            "phase": phase,
            "labels": batch["is_real"][:, 0].float().to(self.device),
        }
        outs.update(self(images))
        if self.num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(input=outs["logits"][:, 0], target=outs["labels"], pos_weight=torch.as_tensor(self.pos_weight, device=self.device))
        else:
            raise NotImplementedError("Only binary classification is implemented!")
        # transfer each tensor to cpu previous to saving them
        for k in outs:
            if isinstance(outs[k], torch.Tensor):
                outs[k] = outs[k].detach().cpu()
        if phase in {"train", "val"}:
            self.log_dict({f"{phase}_{k}": v for k, v in [("loss", loss.detach().cpu()), ("learning_rate", self.optimizers().param_groups[0]["lr"])]}, prog_bar=False, logger=True)
        else:
            self.log_dict({f"{phase}_{k}": v for k, v in [("loss", loss.detach().cpu())]}, prog_bar=False, logger=True)
        # saves the outputs
        self.epoch_outs.append(outs)
        return loss
    
    def _on_start(self):
        with torch.no_grad():
            flops = FlopCountAnalysis(self, torch.randn(1, 3, 224, 224, device=self.device))
            parameters = parameter_count(self)[""]
            self.log_dict({
                "flops": flops.total(),
                "parameters": parameters
                }, prog_bar=True, logger=True)
            
        
    def _on_epoch_start(self):
        self._clear_memory()
        self.epoch_outs = []
    
    def _on_epoch_end(self):
        self._clear_memory()
        with torch.no_grad():
            labels = torch.cat([batch["labels"] for batch in self.epoch_outs], dim=0)
            logits = torch.cat([batch["logits"] for batch in self.epoch_outs], dim=0)[:, 0]
            phases = [phase for batch in self.epoch_outs for phase in [batch["phase"]] * len(batch["labels"])]
            assert len(labels) == len(logits), f"{len(labels)} != {len(logits)}"
            assert len(phases) == len(labels), f"{len(phases)} != {len(labels)}"
            for phase in ["train", "val", "test"]:
                indices_phase = [i for i in range(len(phases)) if phases[i] == phase]
                if len(indices_phase) == 0:
                    continue                
                metrics = {
                    "acc": accuracy(preds=logits[indices_phase], target=labels[indices_phase], task="binary", average="micro"),
                    "auc": auroc(preds=logits[indices_phase], target=labels[indices_phase].long(), task="binary", average="micro"),
                }
                self.log_dict({f"{phase}_{k}": v for k, v in metrics.items() if isinstance(v, (torch.Tensor, int, float))}, prog_bar=True, logger=True)
                    
    def _clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
         
        
if __name__ == "__main__":
    model = EFB4DFR(backbone='resnet56_cifar10',num_classes=2)
    # runs a dummy forward pass to check if the model is working properly
    # model(torch.randn(1, 3, 384, 384))
    model(torch.randn(1, 3, 224, 224))

    # import torchvision.models as models
    # # 如果想要使用預訓練權重，可以傳入 weights 參數
    # model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)


    print(model)

    # import sys
    # sys.path.insert(0, "/ssd6/Roy")
    # from EfficientNetV2.efficientnetv2.efficientnet_v2 import get_efficientnet_v2
    # model_name = "efficientnet_v2_s"
    # pretrained = True 
    # num_classes = 1000
    # model = get_efficientnet_v2(model_name, pretrained, num_classes)

    # with torch.no_grad():
    #     flops = FlopCountAnalysis(model, torch.randn(1, 3, 384, 384))
    #     parameters = parameter_count(model)[""]
    #     print("flops:",flops.total())
    #     print("parameters:",parameters)

