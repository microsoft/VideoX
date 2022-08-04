import torch
import torch.nn.functional as F
def get_padded_mask_and_weight(*args):
    if len(args) == 2:
        mask, conv = args
        masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1, *conv.kernel_size).cuda(),
                                             stride=conv.stride, padding=conv.padding, dilation=conv.dilation))
    elif len(args) == 5:
        mask, k, s, p, d = args
        masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1, k, k).cuda(), stride=s, padding=p, dilation=d))
    else:
        raise NotImplementedError

    masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0] #conv.kernel_size[0] * conv.kernel_size[1]
    padded_mask = masked_weight > 0

    return padded_mask, masked_weight

from .map_conv import MapConv