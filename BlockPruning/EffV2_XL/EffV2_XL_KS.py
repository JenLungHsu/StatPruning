import torch

import sys
sys.path.append('/ssd5/Roy/BlockPruning')
import model_efficient


# def EffV2_XL_KS():
#     finetuned_weights_path = '/ssd5/Roy/BlockPruning/paper2025/ff++c23_20250307_1620_21k_EffV2_XL_21k_/checkpoints/ff++c23_EffV2_XL_epoch=2-train_acc=1.00-val_acc=0.95.ckpt'
#     model = model_efficient.EFB4DFR.load_from_checkpoint(finetuned_weights_path)
#     model = model.base_model

#     for i in range(31, 3, -1):  # 逆向刪除，避免索引錯亂
#         del model.blocks[5][i]
#     for i in range(23, 7, -1):  # 逆向刪除，避免索引錯亂
#         del model.blocks[4][i]
#     for i in range(15, 6, -1):  # 逆向刪除，避免索引錯亂
#         del model.blocks[3][i]
#     del model.blocks[3][4]

#     return model

def EffV2_XL_KS():
    finetuned_weights_path = '/ssd5/Roy/BlockPruning/paper2025/celebdf_20250602_0928_21k_EffV2_XL_21k_/checkpoints/celebdf_EffV2_XL_epoch=3-train_acc=1.00-val_acc=1.00.ckpt'
    model = model_efficient.EFB4DFR.load_from_checkpoint(finetuned_weights_path)
    model = model.base_model

    for i in range(31, 19, -1):  # 逆向刪除，避免索引錯亂
        del model.blocks[5][i]

    del model.blocks[4][23]
    del model.blocks[4][22]
    del model.blocks[4][16]
    del model.blocks[4][15]

    for i in range(15, 9, -1):  # 逆向刪除，避免索引錯亂
        del model.blocks[3][i]
    del model.blocks[3][8]

    return model

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')

    model = EffV2_XL_KS().to(device)
    print(model)
    model(torch.randn(1, 3, 224, 224).to(device))

    from fvcore.nn import FlopCountAnalysis, parameter_count
    with torch.no_grad():
        flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224).to(device))
        parameters = parameter_count(model)[""]
        print("flops:",flops.total())
        print("parameters:",parameters)
