import torch

import sys
sys.path.append('/ssd5/Roy/BlockPruning')
import model_efficient


def EffV2_S_EMD04():
    finetuned_weights_path = '/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250307_1617_21k_EffV2_S_21k_/checkpoints/ff++c23_EffV2_S_epoch=1-train_acc=1.00-val_acc=0.93.ckpt'
    model = model_efficient.EFB4DFR.load_from_checkpoint(finetuned_weights_path)
    model = model.base_model

    del model.blocks[5][14]
    for i in range(12, 0, -1):  # 逆向刪除，避免索引錯亂
        del model.blocks[5][i]

    for i in range(8, 1, -1):  # 逆向刪除，避免索引錯亂
        del model.blocks[4][i]

    for i in range(5, 1, -1):  # 逆向刪除，避免索引錯亂
        del model.blocks[3][i]

    del model.blocks[2][3]
    del model.blocks[2][2]
    
    del model.blocks[1][3]
    del model.blocks[1][2]

    return model

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    device = torch.device('cuda')

    model = EffV2_S_EMD04().to(device)
    print(model)
    model(torch.randn(1, 3, 224, 224).to(device))

    from fvcore.nn import FlopCountAnalysis, parameter_count
    with torch.no_grad():
        flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).to(device))
        parameters = parameter_count(model)[""]
        print("flops:",flops.total())
        print("parameters:",parameters)
