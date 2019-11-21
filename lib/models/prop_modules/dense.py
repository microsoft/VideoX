from torch import nn

class PropMaxPool(nn.Module):
    def __init__(self, cfg):
        super(PropMaxPool, self).__init__()
        num_layers = cfg.NUM_LAYERS
        self.layers = nn.ModuleList(
            [nn.Identity()]
            +[nn.MaxPool1d(2, stride=1) for _ in range(num_layers-1)]
        )
        self.num_layers = num_layers

    def forward(self, x):
        batch_size, hidden_size, num_clips = x.shape
        map_h = x.new_zeros(batch_size, hidden_size, num_clips, num_clips).cuda()
        map_mask = x.new_zeros(batch_size, 1, num_clips, num_clips).cuda()

        for dig_idx, pool in enumerate(self.layers):
            x = pool(x)
            start_idxs = [s_idx for s_idx in range(0, num_clips - dig_idx, 1)]
            end_idxs = [s_idx + dig_idx for s_idx in start_idxs]
            map_h[:, :, start_idxs, end_idxs] = x
            map_mask[:, :, start_idxs, end_idxs] += 1

        return map_h, map_mask