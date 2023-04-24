"""
Encoder modules: we use ViT for the encoder.
"""

from torch import nn
from lib.utils.misc import is_main_process
from lib.models.seqtrack import vit as vit_module




class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):
        super().__init__()
        open_blocks = open_layers[2:]
        open_items = open_layers[0:2]
        for name, parameter in encoder.named_parameters():

            if not train_encoder:
                freeze = True
                for open_block in open_blocks:
                    if open_block in name:
                        freeze = False
                if name in open_items:
                    freeze = False
                if freeze == True:
                    parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !

        self.body = encoder
        self.num_channels = num_channels

    def forward(self, images_list):
        xs = self.body(images_list)
        return xs


class Encoder(EncoderBase):
    """ViT encoder."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 pretrain_type: str,
                 search_size: int,
                 search_number: int,
                 template_size: int,
                 template_number: int,
                 open_layers: list,
                 cfg=None):
        if "vit" in name.lower():
            encoder = getattr(vit_module, name)(pretrained=is_main_process(), pretrain_type=pretrain_type,
                                                       search_size=search_size, template_size=template_size,
                                                       search_number=search_number, template_number=template_number,
                                                       drop_path_rate=cfg.MODEL.ENCODER.DROP_PATH,
                                                       use_checkpoint=cfg.MODEL.ENCODER.USE_CHECKPOINT
                                                      )
            if "_base_" in name:
                num_channels = 768
            elif "_large_" in name:
                num_channels = 1024
            elif "_huge_" in name:
                num_channels = 1280
            else:
                num_channels = 768

        else:
            raise ValueError()
        super().__init__(encoder, train_encoder, open_layers, num_channels)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                      cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.NUMBER,
                      cfg.DATA.TEMPLATE.SIZE, cfg.DATA.TEMPLATE.NUMBER,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    return encoder
