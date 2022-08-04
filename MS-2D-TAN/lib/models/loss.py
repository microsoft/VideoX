import torch
import torch.nn.functional as F

def bce_rescale_loss(scores, masks, targets, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    target_prob = (targets-min_iou)*(1-bias)/(max_iou-min_iou)
    target_prob[target_prob > 0] += bias
    target_prob[target_prob > 1] = 1
    target_prob[target_prob < 0] = 0
    loss = F.binary_cross_entropy(scores[:,:,:target_prob.shape[2]], target_prob, reduction='none') * masks[:,:,:target_prob.shape[2]]
    loss_value = torch.sum(loss) / torch.sum(masks)
    return loss_value