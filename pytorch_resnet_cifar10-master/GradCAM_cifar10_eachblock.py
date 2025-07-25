import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import resnet_cifar100
from PIL import Image
import os
import gc

# 建立儲存資料夾
os.makedirs("gradcam_layers", exist_ok=True)

# 裝置與模型載入
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet_cifar100.__dict__['resnet56']()
# ckpt = torch.load("/ssd5/Roy/pytorch_resnet_cifar10-master/pretrained_models/resnet56.th", map_location=device)
ckpt = torch.load('/ssd5/Roy/pytorch_resnet_cifar10-master/save_resnet56_cifar100/best_model.th')
state_dict = ckpt["state_dict"]
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval().to(device)

# 圖片與轉換
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])
dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
classes = dataset.classes  # e.g. ['airplane', 'automobile', ..., 'truck']

# 取得隨機圖片與標籤
idx = random.randint(0, len(dataset) - 1)
image, label = dataset[idx]
input_tensor = image.unsqueeze(0).to(device)

# 還原成 RGB 圖片
inv_transform = transforms.Normalize(
    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
    std=[1/0.2023, 1/0.1994, 1/0.2010]
)
rgb_img = inv_transform(image).permute(1, 2, 0).numpy()
rgb_img = np.clip(rgb_img, 0, 1)

# 預測結果
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
targets = [ClassifierOutputTarget(predicted_class)]

# 取得所有 block
layer_blocks = model.layer1 + model.layer2 + model.layer3  # 共 27 個 blocks

# 儲存單張與 grid 的圖
cam_images = [rgb_img]  # 第一張放原圖

for i, block in enumerate(layer_blocks):
    try:
        cam = GradCAM(model=model, target_layers=[block])
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        visualization_np = visualization.astype(np.float32) / 255.0

        # 儲存單張圖
        block_path = f"gradcam_layers/block_{i+1:02d}_pred.png"
        Image.fromarray(visualization).save(block_path)

        cam_images.append(visualization_np)

    except AttributeError as e:
        print(f"❌ 第 {i+1} 層 AttributeError：{e}")
    except Exception as e:
        print(f"❌ 第 {i+1} 層其他錯誤：{e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()

# 畫成 grid
n_cols = 7
n_rows = int(np.ceil(len(cam_images) / n_cols))
plt.figure(figsize=(2.5 * n_cols, 2.5 * n_rows))

for i, cam_img in enumerate(cam_images):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.imshow(cam_img)
    if i == 0:
        plt.title(f"Original\nLabel: {classes[label]}\nPred: {classes[predicted_class]}")
    else:
        plt.title(f"Block {i}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("gradcam_layers/grid_all_blocks.png")
plt.show()

print("✅ 所有 Grad-CAM 單張與總覽圖已儲存至 'gradcam_layers/'")
