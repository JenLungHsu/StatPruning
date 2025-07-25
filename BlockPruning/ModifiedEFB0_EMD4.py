import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model 
import os

class ModifiedEFB0_EMD4(nn.Module):
    def __init__(self, original_model):
        super(ModifiedEFB0_EMD4, self).__init__()
        self.init_block = original_model.features.init_block
        self.stage1 = original_model.features.stage1
        
        # self.adapter_stage1_to_stage4 = nn.Sequential(
        #     nn.Conv2d(16, 80, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(80),
        #     nn.ReLU(inplace=True)
        # )


        # 取得原本的 Conv2d 層
        old_conv = original_model.features.stage1.unit1.pw_conv.conv
        # 重新建立一個新的 Conv2d 層，修改 in_channels 為 48，其它參數保持不變
        new_conv = torch.nn.Conv2d(
            in_channels=old_conv.in_channels,  # 修改這裡
            out_channels=80,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None  # 保持 bias 設定
        )
        # 替換舊的 Conv2d 層
        original_model.features.stage1.unit1.pw_conv.conv = new_conv

        old_bn = original_model.features.stage1.unit1.pw_conv.bn
        new_bn = torch.nn.BatchNorm2d(80,
                                      eps=old_bn.eps,
                                      momentum=old_bn.momentum,
                                      affine=old_bn.affine,
                                      track_running_stats=old_bn.track_running_stats)
        original_model.features.stage1.unit1.pw_conv.bn = new_bn



        self.stage4 = nn.Sequential(
            original_model.features.stage4.unit4
        )
        self.stage5 = nn.Sequential(
            original_model.features.stage5.unit1,
            original_model.features.stage5.unit5
        )
        
        # 後面的 final_block 與 final_pool
        self.final_block = original_model.features.final_block
        self.final_pool  = original_model.features.final_pool
        
        # 輸出層保持不變
        self.output = original_model.output

    def forward(self, x):
        x = self.init_block(x)
        x = self.stage1(x)
        # x = self.adapter_stage1_to_stage4(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.final_block(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # 使用方式，例如：
    original_model = ptcv_get_model("efficientnet_b0", pretrained=True)
    # 載入你的 finetuned 權重
    finetuned_weights_path = "/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250219_1054_results_ff++c23_EFB0_test_auc_5epoch/checkpoints/ff++c23_EFB0_epoch=1-train_acc=0.99-val_acc=0.93.ckpt"
    checkpoint = torch.load(finetuned_weights_path)
    original_model.load_state_dict(checkpoint, strict=False)

    # 建立新模型，只保留你想要的部分
    modified_model = ModifiedEFB0_EMD4(original_model)
    print(modified_model)

    input_tensor = torch.randn(1, 3, 224, 224)
    out = modified_model(input_tensor)
    print(out)