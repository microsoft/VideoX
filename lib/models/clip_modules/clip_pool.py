import torch
from torch import nn
import torch.nn.functional as F

class ClipAvgPool(nn.Module):

    def __init__(self, cfg):
        super(ClipAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_size = cfg.HIDDEN_SIZE
        kernel_size = cfg.KERNEL_SIZE
        stride = cfg.STRIDE
        groups = getattr(cfg, 'GROUPS', 1)
        self.vis_conv = nn.Conv1d(input_size*groups, hidden_size*groups, 1, 1, groups=groups)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, visual_input, visual_mask):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.avg_pool(vis_h)
        mask = (F.avg_pool1d(visual_mask, self.avg_pool.kernel_size[0], self.avg_pool.stride[0]) == 1).float()
        vis_h = vis_h * mask
        return vis_h, mask
