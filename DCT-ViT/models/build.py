# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .simmim import build_simmim
from  .dct_vit import DCT_ViT

def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if is_pretrain:
        model = build_simmim(config)
        return model

    if model_type == 'dct_vit':
        model = DCT_ViT(img_size=config.DATA.IMG_SIZE,
                           in_chans=config.MODEL.MY_MODEL_V3.IN_CHANS,
                           num_classes=config.MODEL.NUM_CLASSES,
                           dims=config.MODEL.MY_MODEL_V3.DIMS,
                           depths=config.MODEL.MY_MODEL_V3.DEPTHS,
                           sp_win_size=config.MODEL.MY_MODEL_V3.SP_WIN_SIZE,
                           sp_num_heads=config.MODEL.MY_MODEL_V3.SP_NUM_HEADS,
                           ch_win_size=config.MODEL.MY_MODEL_V3.CH_WIN_SIZE,
                           ch_num_heads=config.MODEL.MY_MODEL_V3.CH_NUM_HEADS,
                           norm_layer=layernorm,
                           drop_path_rate=config.MODEL.MY_MODEL_V3.DROP_PATH_RATE
                           )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
