import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import *

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# === 1. 模型與預訓練權重 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ori = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

model_MW = models.resnet50()
del model_MW.layer2[3]  # 若你有剪枝
# 載入 finetuned 權重
ckpt = torch.load("/ssd5/Roy/pytorch_resnet_cifar10-master/save_resnet50_imagenet/MW/best_model.th", weights_only=True, map_location=device)
model_MW.load_state_dict(ckpt["state_dict"])

model_ori.eval().to(device)
model_MW.eval().to(device)

# === 2. Target layer (最後一層 conv) ===
target_layers_ori = [model_ori.layer4[-1]]
target_layers_MW = [model_MW.layer4[-1]]

# === 3. 圖片處理 ===
### IMAGENET
# img_path = "/ssd5/Roy/train/n02109961/n02109961_3478.JPEG"
# img_path = "/ssd5/Roy/train/n02088632/n02088632_8937.JPEG"
# img_path = "/ssd5/Roy/train/n02088632/n02088632_8172.JPEG"
img_path = "/ssd5/Roy/train/n02088632/n02088632_8913.JPEG"
img_path = "/ssd5/Roy/train/n02088632/n02088632_9790.JPEG"
img_path = "/ssd5/Roy/train/n02088632/n02088632_9950.JPEG"

import os
import random

# 指定資料夾路徑
# img_dir = "/ssd5/Roy/train/n02088632"
# 從資料夾中隨機挑選一張 JPEG 圖片
# img_filename = random.choice([f for f in os.listdir(img_dir) if f.endswith(".JPEG")])
# img_path = os.path.join(img_dir, img_filename)

print(f"👉 隨機選取的圖片路徑：{img_path}")



img = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
input_tensor = transform(img).unsqueeze(0).to(device)

# === 4. 原圖 for 可視化 ===
rgb_img = np.array(img.resize((224, 224))).astype(np.float32) / 255.0

# === 5. Grad-CAM 設定與推論 ===
cam_ori = GradCAM(model=model_ori, target_layers=target_layers_ori)
cam_MW = GradCAM(model=model_MW, target_layers=target_layers_MW)

targets = [ClassifierOutputTarget(243)]  # ImageNet 中 "Husky" 的 index

grayscale_cam_ori = cam_ori(input_tensor=input_tensor, targets=targets)[0]
grayscale_cam_MW = cam_MW(input_tensor=input_tensor, targets=targets)[0]

visualization_ori = show_cam_on_image(rgb_img, grayscale_cam_ori, use_rgb=True)
visualization_MW = show_cam_on_image(rgb_img, grayscale_cam_MW, use_rgb=True)

# === 6. 顯示結果 ===
plt.imshow(rgb_img)
plt.axis('off')
plt.savefig("gradcam_imagenet_bluetick.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_result.png")

plt.imshow(visualization_ori)
plt.axis('off')
plt.savefig("gradcam_imagenet_bluetick_ori.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_result.png")

plt.imshow(visualization_MW)
plt.axis('off')
plt.savefig("gradcam_imagenet_bluetick_MW.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_result.png")

