import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import *
import torchvision
import torchvision.transforms as transforms
import random

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# === 1. 模型與預訓練權重 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import model_efficient
base_model = model_efficient.EFB4DFR.load_from_checkpoint("/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250307_1617_21k_EffV2_S_21k_/checkpoints/ff++c23_EffV2_S_epoch=1-train_acc=1.00-val_acc=0.93.ckpt")
model_ori = base_model.base_model
model_ori.cuda()

# def EffV2_S_KS():
#     finetuned_weights_path = '/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250307_1617_21k_EffV2_S_21k_/checkpoints/ff++c23_EffV2_S_epoch=1-train_acc=1.00-val_acc=0.93.ckpt'
#     model = model_efficient.EFB4DFR.load_from_checkpoint(finetuned_weights_path)
#     model = model.base_model
#     for i in range(11, 0, -1):  # 逆向刪除，避免索引錯亂
#         del model.blocks[5][i]
#     del model.blocks[4][7]
#     del model.blocks[4][4]
#     for i in range(5, 3, -1):  # 逆向刪除，避免索引錯亂
#         del model.blocks[3][i]
#     return model
# model = EffV2_S_KS().to(device)
# ckpt = torch.load('/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250311_1554_S_KS_/checkpoints/ff++c23_EffV2_S_KS_epoch=0-train_acc=0.99-val_acc=0.93.ckpt', map_location="cpu")
# print(ckpt.keys())

model_KS = model_efficient.EFB4DFR.load_from_checkpoint('/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250311_1554_S_KS_/checkpoints/ff++c23_EffV2_S_KS_epoch=0-train_acc=0.99-val_acc=0.93.ckpt')  # 加載權重字典
model_KS = model_KS.base_model

model_ori.eval().to(device)
model_KS.eval().to(device)

# === 2. Target layer (最後一層 conv) ===
target_layers_ori = [model_ori.conv_head]
target_layers_KS = [model_KS.conv_head]

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
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Real_youtube/002/011.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/NeuralTextures/002_006/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/FaceSwap/002_006/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Face2Face/002_006/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Deepfakes/002_006/000.png'

img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Real_youtube/030/010.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/NeuralTextures/030_193/010.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/FaceSwap/030_193/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Face2Face/030_193/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Deepfakes/030_193/020.png'

# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Real_youtube/051/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/NeuralTextures/051_332/009.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/FaceSwap/051_332/009.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Face2Face/051_332/009.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Deepfakes/051_332/000.png'

# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Real_youtube/014/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/NeuralTextures/014_790/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/FaceSwap/014_790/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Face2Face/014_790/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Deepfakes/014_790/000.png'

# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Real_youtube/064/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/NeuralTextures/064_991/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/FaceSwap/064_991/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Face2Face/064_991/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Deepfakes/064_991/000.png'

# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Real_youtube/000/003.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/NeuralTextures/000_003/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/FaceSwap/000_003/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Face2Face/000_003/001.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Deepfakes/000_003/000.png'

# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Real_youtube/001/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/NeuralTextures/001_870/002.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/FaceSwap/001_870/001.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Face2Face/001_870/000.png'
# img_path = '/ssd2/DeepFakes_may/FF++/c23_crop_face/train/Deepfakes/001_870/002.png'
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
cam_KS = GradCAM(model=model_KS, target_layers=target_layers_KS)

# 取得模型預測的類別（也可以用 label 替代）
with torch.no_grad():
    output = model_ori(input_tensor)
    predicted_class = output.argmax(dim=1).item()

targets = [ClassifierOutputTarget(predicted_class)]

# CAM
grayscale_cam_ori = cam_ori(input_tensor=input_tensor, targets=targets)[0]
grayscale_cam_KS = cam_KS(input_tensor=input_tensor, targets=targets)[0]

visualization_ori = show_cam_on_image(rgb_img, grayscale_cam_ori, use_rgb=True)
visualization_KS = show_cam_on_image(rgb_img, grayscale_cam_KS, use_rgb=True)

# === 6. 顯示結果 ===
plt.imshow(rgb_img)
plt.axis('off')
plt.savefig("gradcam_ff++_REAL.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_ff++_result.png")

plt.imshow(visualization_ori)
plt.axis('off')
plt.savefig("gradcam_ff++_ori_REAL.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_ff++_result.png")

plt.imshow(visualization_KS)
plt.axis('off')
plt.savefig("gradcam_ff++_KS_REAL.png", bbox_inches='tight', pad_inches=0)
print("✔️ Grad-CAM 圖片已儲存為 gradcam_ff++_result.png")

