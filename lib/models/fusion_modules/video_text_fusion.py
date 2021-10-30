import torch
from torch import nn
import torch.nn.functional as F

class MulNormFusion(nn.Module):

    def __init__(self, cfg):
        super(MulNormFusion, self).__init__()
        self.cfg = cfg
        self.textual_encoder = getattr(nn, cfg.TXT_ENCODER.NAME)(
            cfg.TXT_INPUT_SIZE, cfg.TXT_HIDDEN_SIZE//2 if cfg.TXT_ENCODER.BIDIRECTIONAL else cfg.TXT_HIDDEN_SIZE,
            num_layers=cfg.TXT_ENCODER.NUM_LAYERS, bidirectional=cfg.TXT_ENCODER.BIDIRECTIONAL, batch_first=True
        )
        self.tex_linear = nn.Linear(cfg.TXT_HIDDEN_SIZE, cfg.HIDDEN_SIZE)
        self.vis_conv = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)

    def forward(self, textual_input, textual_mask, map_h, map_mask):
        self.textual_encoder.flatten_parameters()
        txt_h = self.textual_encoder(textual_input)[0] * textual_mask
        txt_h = torch.sum(txt_h, dim=1)/torch.sum(textual_mask, dim=1)
        txt_h = self.tex_linear(txt_h)[:,:,None,None]
        map_h = self.vis_conv(map_h)
        fused_h = F.normalize(txt_h * map_h) * map_mask

        return fused_h, map_mask