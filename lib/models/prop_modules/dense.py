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
        map_h = x.new_zeros(batch_size, hidden_size, num_clips, self.num_layers).cuda()
        map_mask = x.new_zeros(batch_size, 1, num_clips, self.num_layers).cuda()

        for i, pool in enumerate(self.layers):
            x = pool(x)
            map_h[:, :, :num_clips-i, i] = x
            map_mask[:, :, :num_clips-i, i] = 1

        return map_h, map_mask

class PropConv(nn.Module):
    def __init__(self, cfg):
        super(PropConv, self).__init__()
        self.cfg = cfg
        self.hidden_size = cfg.HIDDEN_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.layers = nn.ModuleList(
            [nn.Conv1d(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE, 1,1)]
            +[nn.Conv1d(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE, 2,1) for _ in range(1, cfg.NUM_LAYERS)]
        )

    def forward(self, x):

        batch_size, hidden_size, num_clips = x.shape
        map_h = x.new_zeros(batch_size, hidden_size, num_clips, self.num_layers).cuda()
        map_mask = x.new_zeros(batch_size, 1, num_clips, self.num_layers).cuda()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            map_h[:, :, :num_clips - i, i] = x
            map_mask[:, :, :num_clips - i, i] = 1
        return map_h, map_mask

class PropConvTanh(PropConv):
    def __init__(self, cfg):
        super(PropConvTanh, self).__init__(cfg)
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(self.hidden_size, self.hidden_size, 1,1), nn.Tanh())]
            +[nn.Sequential(nn.Conv1d(self.hidden_size, self.hidden_size, 2,1), nn.Tanh()) for _ in range(1, self.num_layers)]
        )

class PropConvTanhBN(PropConv):
    def __init__(self, cfg):
        super(PropConvTanhBN, self).__init__(cfg)
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(self.hidden_size, self.hidden_size, 1,1), nn.BatchNorm1d(self.hidden_size), nn.Tanh())]
            +[nn.Sequential(nn.Conv1d(self.hidden_size, self.hidden_size, 2,1), nn.BatchNorm1d(self.hidden_size), nn.Tanh()) for _ in range(1, self.num_layers)]
        )