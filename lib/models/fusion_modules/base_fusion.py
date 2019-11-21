import torch
from torch import nn
import torch.nn.functional as F

class BaseFusion(nn.Module):

    def __init__(self, cfg):
        super(BaseFusion, self).__init__()
        self.cfg = cfg
        hidden_size = cfg.HIDDEN_SIZE
        txt_input_size = cfg.TXT_INPUT_SIZE
        txt_hidden_size = cfg.TXT_HIDDEN_SIZE
        self.textual_encoder = nn.LSTM(txt_input_size, txt_hidden_size//2 if cfg.LSTM.BIDIRECTIONAL else txt_hidden_size,
                                       num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL, batch_first=True)
        self.tex_linear = nn.Linear(txt_hidden_size, hidden_size)
        self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)

    def forward(self, textual_input, textual_mask, map_h, map_mask):
        self.textual_encoder.flatten_parameters()
        txt_h = self.textual_encoder(textual_input)[0] * textual_mask
        txt_h = torch.stack([txt_h[i][torch.sum(mask).long() - 1] for i, mask in enumerate(textual_mask)])
        txt_h = self.tex_linear(txt_h)[:,:,None,None]
        map_h = self.vis_conv(map_h)
        fused_h = F.normalize(txt_h * map_h) * map_mask
        return fused_h

