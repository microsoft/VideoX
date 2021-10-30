from torch import nn

class SparsePropConv(nn.Module):
    def __init__(self, cfg):
        super(SparsePropConv, self).__init__()
        self.num_scale_layers = cfg.NUM_LAYERS
        self.hidden_size = cfg.HIDDEN_SIZE

        self.layers = nn.ModuleList()

        for scale_idx, num_layer in enumerate(self.num_scale_layers):
            scale_layers = nn.ModuleList()
            first_layer = nn.Conv1d(self.hidden_size, self.hidden_size, 1,1) if scale_idx == 0 else nn.Conv1d(self.hidden_size, self.hidden_size, 3, 2)
            rest_layers = [nn.Conv1d(self.hidden_size, self.hidden_size, 2,1) for _ in range(1, num_layer)]
            scale_layers.extend([first_layer]+rest_layers)
            self.layers.append(scale_layers)

    def forward(self, x, mask):
        batch_size, hidden_size, num_scale_clips = x.shape
        num_anchors = len(self.layers[0])*2**(len(self.layers)-1)
        map_h = x.new_zeros(batch_size, hidden_size, num_scale_clips, num_anchors)
        map_mask = x.new_zeros(batch_size, 1, num_scale_clips, num_anchors)

        anchor_idx = 0
        for scale_idx, scale_layers in enumerate(self.layers):
            dilation = 2**scale_idx
            for i, layer in enumerate(scale_layers):
                x = layer(x)
                map_h[:, :, :x.shape[-1]*dilation:dilation, anchor_idx] = x
                map_mask[:, :, :x.shape[-1]*dilation:dilation, anchor_idx] = 1
                anchor_idx += dilation
            anchor_idx += 2**(scale_idx+1) - dilation

        return [map_h], [map_mask]

class SparsePropConvTanhBN(SparsePropConv):
    def __init__(self, cfg):
        super(SparsePropConvTanhBN, self).__init__(cfg)
        self.layers = nn.ModuleList()

        for scale_idx, num_layer in enumerate(self.num_scale_layers):
            scale_layers = nn.ModuleList()
            first_layer = nn.Sequential(
                nn.Conv1d(self.hidden_size, self.hidden_size, 1, 1),
                nn.BatchNorm1d(self.hidden_size),
                nn.Tanh()
            ) if scale_idx == 0 else nn.Sequential(
                nn.Conv1d(self.hidden_size, self.hidden_size, 3, 2),
                nn.BatchNorm1d(self.hidden_size),
                nn.Tanh()
            )
            rest_layers = [nn.Sequential(
                nn.Conv1d(self.hidden_size, self.hidden_size, 2, 1),
                nn.BatchNorm1d(self.hidden_size),
                nn.Tanh()
            ) for _ in range(1, num_layer)]
            scale_layers.extend([first_layer] + rest_layers)
            self.layers.append(scale_layers)

