import torch
from torch import nn

class ProposalHead(nn.Module):
    def __init__(self, cfg):
        super(ProposalHead, self).__init__()
        self.cfg = cfg
        groups = getattr(cfg, 'GROUPS', 1)
        self.predictor = nn.Conv2d(cfg.INPUT_SIZE*groups, getattr(cfg, 'OUTPUT_SIZE', groups), 1, 1, groups=groups)

    def forward(self, vis_input, mask):
        output = self.predictor(vis_input)*mask
        return torch.sigmoid(output), mask