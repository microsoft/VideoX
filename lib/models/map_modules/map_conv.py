from models.map_modules import get_padded_mask_and_weight, get_avg_padded_mask_and_weight
from .blocks import *

class Identity(nn.Module):

    def __init__(self, cfg):
        super(Identity, self).__init__()

    def forward(self, x, mask):
        return x, mask

class Conv(nn.Module):

    def __init__(self, cfg):
        super(Conv, self).__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList()
        assert len(cfg.HIDDEN_SIZES) == len(cfg.KERNEL_SIZES) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.STRIDES) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.PADDINGS) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.DILATIONS)
        groups = cfg.GROUPS if 'GROUPS' in cfg else [1 for _ in cfg.HIDDEN_SIZES]
        channel_sizes = [cfg.INPUT_SIZE]+cfg.HIDDEN_SIZES
        for i, (k, s, p, d, g) in enumerate(zip(cfg.KERNEL_SIZES, cfg.STRIDES, cfg.PADDINGS, cfg.DILATIONS, groups)):
            self.convs.append(nn.Conv2d(channel_sizes[i]*g, channel_sizes[i+1]*g, k, s, p, d, groups=g))

    def forward(self, x, mask):
        padded_mask = mask
        for i, conv in enumerate(self.convs):
            x = torch.relu(conv(x))
        return x, mask

class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList()
        assert len(cfg.HIDDEN_SIZES) == len(cfg.KERNEL_SIZES) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.STRIDES) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.PADDINGS) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.DILATIONS)
        channel_sizes = [cfg.INPUT_SIZE]+cfg.HIDDEN_SIZES
        for i, (k, s, p, d) in enumerate(zip(cfg.KERNEL_SIZES, cfg.STRIDES, cfg.PADDINGS, cfg.DILATIONS)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], k, s, p, d))

    def forward(self, x, mask):
        padded_mask = mask
        for i, conv in enumerate(self.convs):
            x = torch.relu(conv(x))
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, conv)
            x = x * masked_weight
        return x, mask

class MapConvAvg(MapConv):
    def forward(self, x, mask):
        padded_mask = mask
        for i, conv in enumerate(self.convs):
            x = torch.relu(conv(x))
            padded_mask, masked_weight = get_avg_padded_mask_and_weight(padded_mask, conv)
            x = x * masked_weight
        return x, mask

class MapGatedConv(nn.Module):
    def __init__(self, cfg):
        super(MapGatedConv, self).__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList()
        groups = getattr(cfg, 'GROUPS', [1 for _ in cfg.HIDDEN_SIZES])
        assert len(cfg.HIDDEN_SIZES) == len(cfg.KERNEL_SIZES) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.STRIDES) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.PADDINGS) \
               and len(cfg.HIDDEN_SIZES) == len(cfg.DILATIONS) \
               and len(cfg.HIDDEN_SIZES) == len(groups)

        in_channels = [cfg.INPUT_SIZE] + cfg.HIDDEN_SIZES[:-1]
        out_channels = cfg.HIDDEN_SIZES

        for i, (k, s, p, d, g) in enumerate(zip(cfg.KERNEL_SIZES, cfg.STRIDES, cfg.PADDINGS, cfg.DILATIONS, groups)):
            self.blocks.append(nn.ModuleList([
                nn.Conv2d(in_channels[i]*g, out_channels[i]*g, k, s, p, d, g),
                nn.Conv2d(in_channels[i]*g, g, k, s, p, d, g),
            ]))

    def forward(self, x, mask, prev_hs=None, return_all=False):
        xs = []
        for i, block in enumerate(self.blocks):
            gate = torch.sigmoid(block[1](x))
            x = torch.relu(block[0](x))
            in_channel = x.shape[1] // gate.shape[1]
            tmp_x = []
            for xx, gg in zip(x.split(in_channel, dim=1), gate.split(1, dim=1)):
                tmp_x.append(xx*gg)
            x = torch.cat(tmp_x, dim=1)
            if prev_hs is not None:
                x[:, :, :prev_hs[i].shape[2]*2:2] += prev_hs[i]
                x[:, :, 1:prev_hs[i].shape[2]*2:2] += prev_hs[i]
            xs.append(x)
        if return_all:
            return xs, mask
        else:
            return x, mask