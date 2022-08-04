from torch import nn
import torch.nn.functional as F

class MultiScalePropMaxPool(nn.Module):
    def __init__(self, cfg):
        super(MultiScalePropMaxPool, self).__init__()
        self.num_scale_layers = cfg.NUM_LAYERS

        self.layers = nn.ModuleList()

        for scale_idx, num_layer in enumerate(self.num_scale_layers):
            scale_layers = nn.ModuleList()
            first_layer = nn.MaxPool1d(1,1) if scale_idx == 0 else nn.MaxPool1d(3,2)
            rest_layers = [nn.MaxPool1d(2,1) for _ in range(1, num_layer)]
            scale_layers.extend([first_layer]+rest_layers)
            self.layers.append(scale_layers)

    def forward(self, x, mask):
        map_h_list = []
        map_mask_list = []

        batch_size, hidden_size, map_size = x.shape
        map_h = x.new_zeros(batch_size, hidden_size, map_size, len(self.layers[0]))
        map_mask = x.new_zeros(batch_size, 1, map_size, len(self.layers[0]))
        for i, layer in enumerate(self.layers[0]):
            mask = (F.avg_pool1d(mask, 1 if i == 0 else 2, 1) == 1).float()
            x = layer(x)*mask
            map_h[:, :, :map_size - i, i] = x
            map_mask[:, :, :map_size - i, i] = mask[...,:map_size-i:1]
        map_h_list.append(map_h)
        map_mask_list.append(map_mask)

        for scale_idx, scale_layers in enumerate(self.layers[1:]):
            dilation = 2**(scale_idx+1)
            num_scale_clips = map_size//dilation
            num_scale_anchors = 2*len(scale_layers)
            map_h = x.new_zeros(batch_size, hidden_size, num_scale_clips, num_scale_anchors)
            map_mask = x.new_zeros(batch_size, 1, num_scale_clips, num_scale_anchors)
            map_h[...,0:num_scale_anchors//2] = map_h_list[-1][:,:,:num_scale_clips*2:2,1::2]
            map_mask[...,0:num_scale_anchors//2] = map_mask_list[-1][:,:,:num_scale_clips*2:2,1::2]
            for i, layer in enumerate(scale_layers):
                mask = (F.avg_pool1d(mask, 3 if i == 0 else 2, 2 if i==0 else 1) == 1).float()
                x = layer(x)*mask
                map_h[:, :, :x.shape[-1], i+num_scale_anchors//2] = x
                map_mask[:, :, :x.shape[-1], i+num_scale_anchors//2] = mask[...,:x.shape[-1]]
            map_h_list.append(map_h)
            map_mask_list.append(map_mask)
        return map_h_list, map_mask_list

class MultiScalePropConv(MultiScalePropMaxPool):
    def __init__(self, cfg):
        super(MultiScalePropConv, self).__init__(cfg)
        self.num_layers = cfg.NUM_LAYERS
        self.hidden_size = cfg.HIDDEN_SIZE

        self.layers = nn.ModuleList()

        for scale_idx, num_layer in enumerate(self.num_layers):
            scale_layers = nn.ModuleList()
            first_layer = nn.Conv1d(self.hidden_size, self.hidden_size, 1,1) if scale_idx == 0 else nn.Conv1d(self.hidden_size, self.hidden_size, 3, 2)
            rest_layers = [nn.Conv1d(self.hidden_size, self.hidden_size, 2,1) for _ in range(1, num_layer)]
            scale_layers.extend([first_layer]+rest_layers)
            self.layers.append(scale_layers)

class MultiScalePropConvTanhBN(MultiScalePropConv):
    def __init__(self, cfg):
        super(MultiScalePropConvTanhBN, self).__init__(cfg)
        self.layers = nn.ModuleList()
        groups = getattr(cfg, 'GROUPS', 1)
        for scale_idx, num_layer in enumerate(self.num_layers):
            scale_layers = nn.ModuleList()
            first_layer = nn.Sequential(
                nn.Conv1d(self.hidden_size*groups, self.hidden_size*groups, 1, 1, groups=groups),
                nn.BatchNorm1d(self.hidden_size*groups),
                nn.Tanh()
            ) if scale_idx == 0 else nn.Sequential(
                nn.Conv1d(self.hidden_size*groups, self.hidden_size*groups, 3, 2, groups=groups),
                nn.BatchNorm1d(self.hidden_size*groups),
                nn.Tanh()
            )
            rest_layers = [nn.Sequential(
                nn.Conv1d(self.hidden_size*groups, self.hidden_size*groups, 2, 1, groups=groups),
                nn.BatchNorm1d(self.hidden_size*groups),
                nn.Tanh()
            ) for _ in range(1, num_layer)]
            scale_layers.extend([first_layer] + rest_layers)
            self.layers.append(scale_layers)