import torch

import sys
sys.path.append('/ssd5/Roy/BlockPruning')
import model_efficient


def EffV2_XL_RedCircle():
    finetuned_weights_path = '/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250307_1620_21k_EffV2_XL_21k_/checkpoints/ff++c23_EffV2_XL_epoch=2-train_acc=1.00-val_acc=0.95.ckpt'
    model = model_efficient.EFB4DFR.load_from_checkpoint(finetuned_weights_path)
    model = model.base_model

    for i in range(7, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.blocks[6][i]

    for i in range(31, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.blocks[5][i]

    for i in range(23, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.blocks[4][i]

    for i in range(15, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.blocks[3][i]

    for i in range(7, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.blocks[2][i]

    for i in range(7, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.blocks[1][i]

    return model

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    device = torch.device('cuda')

    model = EffV2_XL_RedCircle().to(device)
    print(model)
    model(torch.randn(1, 3, 224, 224).to(device))

    from fvcore.nn import FlopCountAnalysis, parameter_count
    with torch.no_grad():
        flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).to(device))
        parameters = parameter_count(model)[""]
        print("flops:",flops.total())
        print("parameters:",parameters)
