import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import *
import resnet_cifar10
import torchvision
import torchvision.transforms as transforms
import random

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# === 1. 模型與預訓練權重 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ori = resnet_cifar10.__dict__['resnet110']()
ckpt = torch.load("/ssd5/Roy/pytorch_resnet_cifar10-master/pretrained_models/resnet110.th", weights_only=True, map_location=device)
state_dict = ckpt["state_dict"]
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model_ori.load_state_dict(new_state_dict)
model_ori.cuda()

model_MW = resnet_cifar10.__dict__['resnet110_MW']()
ckpt = torch.load("/ssd5/Roy/pytorch_resnet_cifar10-master/save_resnet110/MW/best_model.th", weights_only=True, map_location=device)
state_dict = ckpt["state_dict"]
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model_MW.load_state_dict(new_state_dict)

model_ori.eval().to(device)
model_MW.eval().to(device)

# === 2. Target layer (最後一層 conv) ===
target_layers_ori = [model_ori.layer3[-1]]
target_layers_MW = [model_MW.layer3[-1]]

# === 3. 圖片處理 ===

# 轉換方式：將圖片轉為 Tensor 並正規化
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

# 載入 CIFAR-10 訓練資料集
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 隨機挑一張圖片
idx = random.randint(0, len(dataset) - 1)
image, label = dataset[idx]
input_tensor = image.unsqueeze(0).to(device)

# 印出圖像資訊
classes = dataset.classes
print(f"🖼️ 隨機選取 CIFAR-10 圖片索引: {idx}")
print(f"🏷️ 圖片 Ground Truth 類別: {label} ({classes[label]})")

# 反 normalize + to numpy for overlay
inv_transform = transforms.Normalize(
    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
    std=[1/0.2023, 1/0.1994, 1/0.2010]
)
rgb_img = inv_transform(image).permute(1, 2, 0).numpy()
rgb_img = np.clip(rgb_img, 0, 1)

# === 5. Grad-CAM 設定與推論 ===
cam_ori = GradCAM(model=model_ori, target_layers=target_layers_ori)
cam_MW = GradCAM(model=model_MW, target_layers=target_layers_MW)

# 取得模型預測的類別（也可以用 label 替代）
with torch.no_grad():
    output = model_ori(input_tensor)
    predicted_class = output.argmax(dim=1).item()

targets = [ClassifierOutputTarget(predicted_class)]

# CAM
grayscale_cam_ori = cam_ori(input_tensor=input_tensor, targets=targets)[0]
grayscale_cam_MW = cam_MW(input_tensor=input_tensor, targets=targets)[0]

visualization_ori = show_cam_on_image(rgb_img, grayscale_cam_ori, use_rgb=True)
visualization_MW = show_cam_on_image(rgb_img, grayscale_cam_MW, use_rgb=True)

# === 6. 顯示結果 ===
plt.imshow(rgb_img)
plt.axis('off')
plt.savefig("gradcam_cifar10_R110.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_result.png")

plt.imshow(visualization_ori)
plt.axis('off')
plt.savefig("gradcam_cifar10_R110_ori.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_result.png")

plt.imshow(visualization_MW)
plt.axis('off')
plt.savefig("gradcam_cifar10_R110_MW.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_result.png")

