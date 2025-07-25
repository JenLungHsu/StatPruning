import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model 
import os

class ModifiedEFB4(nn.Module):
    def __init__(self, original_model):
        super(ModifiedEFB4, self).__init__()
        # 保留 features 中的 init_block
        self.init_block = original_model.features.init_block
        
        # stage1：只保留 unit2（刪除 unit1）
        self.stage1 = original_model.features.stage1
        
        # stage2：保留全部
        self.stage2 = original_model.features.stage2
        
        # stage3：只保留 unit1、unit2、unit4（刪除 unit3）
        self.stage3 = nn.Sequential(
            original_model.features.stage3.unit1,
            original_model.features.stage3.unit2,
            original_model.features.stage3.unit4
        )
        
        # stage4：只保留 unit1、unit4（假設 stage4 有 12 個 unit，我們只挑選 unit1 與 unit4）
        self.stage4 = nn.Sequential(
            original_model.features.stage4.unit1,
            original_model.features.stage4.unit4
        )

        # 在 stage4 與 stage5 之間插入 adapter，將 112 channel 轉換成 160 channel
        self.adapter_stage4_to_stage5 = nn.Sequential(
            nn.Conv2d(112, 160, kernel_size=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True)
        )

        # stage5：只保留 unit1、unit9（假設 stage5 有 10 個 unit）
        self.stage5 = nn.Sequential(
            original_model.features.stage5.unit1,
            original_model.features.stage5.unit9
        )
        
        # 後面的 final_block 與 final_pool
        self.final_block = original_model.features.final_block
        self.final_pool  = original_model.features.final_pool
        
        # 輸出層保持不變
        self.output = original_model.output

    def forward(self, x):
        x = self.init_block(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.adapter_stage4_to_stage5(x)
        x = self.stage5(x)
        x = self.final_block(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # 使用方式，例如：
    original_model = ptcv_get_model("efficientnet_b4c", pretrained=True)
    # 載入你的 finetuned 權重
    finetuned_weights_path = "/ssd6/Roy/Few-Shot-Forgery-Detection/ckpt/2_EfficientNetB4_pytorch.pkl"
    checkpoint = torch.load(finetuned_weights_path)
    original_model.load_state_dict(checkpoint, strict=False)

    # 建立新模型，只保留你想要的部分
    modified_model = ModifiedEFB4(original_model)
    print(modified_model)

    input_tensor = torch.randn(1, 3, 224, 224)
    out = modified_model(input_tensor)
    print(out)