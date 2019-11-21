from torch import nn

class SparsePropMaxPool(nn.Module):
    def __init__(self, cfg):
        super(SparsePropMaxPool, self).__init__()
        self.num_scale_layers = cfg.NUM_SCALE_LAYERS

        self.layers = nn.ModuleList()

        for scale_idx, num_layer in enumerate(self.num_scale_layers):
            scale_layers = nn.ModuleList()
            first_layer = nn.MaxPool1d(1,1) if scale_idx == 0 else nn.MaxPool1d(3,2)
            rest_layers = [nn.MaxPool1d(2,1) for _ in range(1, num_layer)]
            scale_layers.extend([first_layer]+rest_layers)
            self.layers.append(scale_layers)

    def forward(self, x):
        map_h_list = []
        map_mask_list = []

        for scale_idx, scale_layers in enumerate(self.layers):
            batch_size, hidden_size, num_scale_clips = x.shape
            num_scale_clips = num_scale_clips//scale_layers[0].stride
            map_h = x.new_zeros(batch_size, hidden_size, num_scale_clips, num_scale_clips)
            map_mask = x.new_zeros(batch_size, 1, num_scale_clips, num_scale_clips)
            for i, layer in enumerate(scale_layers):
                try:
                    x = layer(x)
                except:
                    pass
                scale_s_idxs = list(range(0, num_scale_clips - i, 1))
                scale_e_idxs = [s_idx + i for s_idx in scale_s_idxs]
                map_h[:,:,scale_s_idxs, scale_e_idxs] = x
                map_mask[:,:,scale_s_idxs, scale_e_idxs] = 1
            map_h_list.append(map_h)
            map_mask_list.append(map_mask)

        ori_map_h, ori_map_mask = self.recover_to_original_map(map_h_list, map_mask_list)
        return ori_map_h, ori_map_mask

    def recover_to_original_map(self, h_list, mask_list):
        # resize to original scale
        batch_size, hidden_size, ori_num_clips, _ = h_list[0].shape

        ori_map_h = h_list[0].new_zeros(batch_size, hidden_size, ori_num_clips, ori_num_clips)
        ori_map_mask = mask_list[0].new_zeros(batch_size, 1, ori_num_clips, ori_num_clips)
        acum_layers = 0
        stride = 1
        for scale_layers, h, mask in zip(self.layers, h_list, mask_list):
            num_scale_clips = h.shape[-1]
            for i, layer in enumerate(scale_layers):
                stride = stride * layer.stride
                scale_s_idxs = list(range(0,num_scale_clips-i,1))
                scale_e_idxs = [s_idx+i for s_idx in scale_s_idxs]
                ori_s_idxs = list(range(0,ori_num_clips-acum_layers-i*stride,stride))
                ori_e_idxs = [s_idx+acum_layers+i*stride for s_idx in ori_s_idxs]
                ori_map_h[:,:, ori_s_idxs, ori_e_idxs] = h[:,:, scale_s_idxs, scale_e_idxs]
                ori_map_mask[:,:, ori_s_idxs, ori_e_idxs] = 1

            acum_layers += stride * (len(scale_layers)+1)

        return ori_map_h, ori_map_mask

class SparsePropConv(nn.Module):
    def __init__(self, cfg):
        super(SparsePropConv, self).__init__()
        self.num_scale_layers = cfg.NUM_SCALE_LAYERS
        self.hidden_size = cfg.HIDDEN_SIZE

        self.layers = nn.ModuleList()

        for scale_idx, num_layer in enumerate(self.num_scale_layers):
            scale_layers = nn.ModuleList()
            first_layer = nn.Conv1d(self.hidden_size, self.hidden_size, 1,1) if scale_idx == 0 else nn.Conv1d(self.hidden_size, self.hidden_size, 3, 2)
            rest_layers = [nn.Conv1d(self.hidden_size, self.hidden_size, 2,1) for _ in range(1, num_layer)]
            scale_layers.extend([first_layer]+rest_layers)
            self.layers.append(scale_layers)

    def forward(self, x):
        map_h_list = []
        map_mask_list = []

        for scale_idx, scale_layers in enumerate(self.layers):
            batch_size, hidden_size, num_scale_clips = x.shape
            num_scale_clips = num_scale_clips//scale_layers[0].stride[0]
            map_h = x.new_zeros(batch_size, hidden_size, num_scale_clips, num_scale_clips)
            map_mask = x.new_zeros(batch_size, 1, num_scale_clips, num_scale_clips)
            for i, layer in enumerate(scale_layers):
                x = layer(x)
                scale_s_idxs = list(range(0, num_scale_clips - i, 1))
                scale_e_idxs = [s_idx + i for s_idx in scale_s_idxs]
                map_h[:,:,scale_s_idxs, scale_e_idxs] = x
                map_mask[:,:,scale_s_idxs, scale_e_idxs] = 1
            map_h_list.append(map_h)
            map_mask_list.append(map_mask)


        ori_map_h, ori_map_mask = self.recover_to_original_map(map_h_list, map_mask_list)

        return ori_map_h, ori_map_mask

    def recover_to_original_map(self, h_list, mask_list):
        # resize to original scale
        batch_size, hidden_size, ori_num_clips, _ = h_list[0].shape

        ori_map_h = h_list[0].new_zeros(batch_size, hidden_size, ori_num_clips, ori_num_clips)
        ori_map_mask = mask_list[0].new_zeros(batch_size, 1, ori_num_clips, ori_num_clips)
        acum_layers = 0
        stride = 1
        for scale_layers, h, mask in zip(self.layers, h_list, mask_list):
            num_scale_clips = h.shape[-1]
            for i, layer in enumerate(scale_layers):
                stride = stride * layer.stride[0]
                scale_s_idxs = list(range(0,num_scale_clips-i,1))
                scale_e_idxs = [s_idx+i for s_idx in scale_s_idxs]
                ori_s_idxs = list(range(0,ori_num_clips-acum_layers-i*stride,stride))
                ori_e_idxs = [s_idx+acum_layers+i*stride for s_idx in ori_s_idxs]
                ori_map_h[:,:, ori_s_idxs, ori_e_idxs] = h[:,:, scale_s_idxs, scale_e_idxs]
                ori_map_mask[:,:, ori_s_idxs, ori_e_idxs] = 1

            acum_layers += stride * (len(scale_layers)+1)

        return ori_map_h, ori_map_mask