import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.models import *
import torchvision
import torchvision.transforms as transforms
import random


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# === 1. 模型與預訓練權重 ===
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


import model_efficient
base_model = model_efficient.EFB4DFR.load_from_checkpoint("/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250307_1618_21k_EffV2_M_21k_/checkpoints/ff++c23_EffV2_M_epoch=3-train_acc=1.00-val_acc=0.94.ckpt")
model_ori = base_model.base_model

model_MW = model_efficient.EFB4DFR.load_from_checkpoint('/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250311_1605_M_MW_/checkpoints/ff++c23_EffV2_M_MW_epoch=1-train_acc=0.99-val_acc=0.92.ckpt')  # 加載權重字典
model_MW = model_MW.base_model

model_ori.eval().to(device)
model_MW.eval().to(device)

# === 2. Target layer (最後一層 conv) ===
target_layers_ori = [model_ori.conv_head]
target_layers_MW = [model_MW.conv_head]

# === 3. 圖片處理 ===

# 轉換方式：將圖片轉為 Tensor 並正規化
# transform = transforms.Compose([
#     transforms.Resize(224 + 224 // 8, interpolation=transforms.InterpolationMode.BILINEAR),
#     transforms.ToTensor(),
#     transforms.CenterCrop(224),
# ])


# from BlockPruning.ffpp_dataset import FaceForensicsDataset
# train_dataset = FaceForensicsDataset(
#     dataset_path='/ssd2/DeepFakes_may/FF++/c23_crop_face',
#     split="train",
#     resolution=224,
# )
# val_dataset = FaceForensicsDataset(
#     dataset_path='/ssd2/DeepFakes_may/FF++/c23_crop_face',
#     split="test", #val 
#     resolution=224,
# )
# dataset = train_dataset


from PIL import Image

# === 讀入自己的圖片 ===
####### Brad Pitt
# img_path = '/ssd2/DeepFakes_may/celeb-df_crop_face/Celeb-real/train/id1_0001/001.png'
# img_path = '/ssd2/DeepFakes_may/celeb-df_crop_face/Celeb-synthesis/train/id1_id4_0001/001.png'

####### Chris Evans
# img_path = '/ssd2/DeepFakes_may/celeb-df_crop_face/Celeb-real/train/id4_0007/200.png'
# img_path = '/ssd2/DeepFakes_may/celeb-df_crop_face/Celeb-synthesis/train/id4_id1_0007/200.png'

####### Scarlett
# img_path = '/ssd2/DeepFakes_may/celeb-df_crop_face/Celeb-real/train/id7_0004/000.png'
# img_path = '/ssd2/DeepFakes_may/celeb-df_crop_face/Celeb-synthesis/val/id7_id13_0004/000.png'

####### Angelina Jolie
# img_path = '/ssd2/DeepFakes_may/celeb-df_crop_face/Celeb-real/test/id13_0012/010.png'
img_path = '/ssd2/DeepFakes_may/celeb-df_crop_face/Celeb-synthesis/train/id13_id7_0012/010.png'


img = Image.open(img_path).convert('RGB')

# 同樣的 transform（和你訓練時保持一致）
transform = transforms.Compose([
    transforms.Resize(224 + 224 // 8, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

image = transform(img)
input_tensor = image.unsqueeze(0).to(device)

# 原圖給 Grad-CAM 疊圖用（要 clip 成 [0, 1]）
rgb_img = np.array(img.resize((224, 224))).astype(np.float32) / 255.0



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
plt.savefig("gradcam_celeb.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_ff++_result.png")

plt.imshow(visualization_ori)
plt.axis('off')
plt.savefig("gradcam_celeb_ori.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_ff++_result.png")

plt.imshow(visualization_MW)
plt.axis('off')
plt.savefig("gradcam_celeb_MW.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_ff++_result.png")

# === 7. 合併三張圖橫向輸出 ===
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 可依圖大小微調

titles = ["Baseline Grad-CAM", "Original Image", "MW Grad-CAM"]
images = [visualization_ori, rgb_img, visualization_MW]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig("gradcam_celeb_combined.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 合併圖已儲存為 gradcam_celeb_combined.png")