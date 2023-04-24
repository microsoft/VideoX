import torch

def get_jittered_box(boxes):
    """ Jitter the input box
    args:
        box - input bounding box
    returns:
        torch.Tensor - jittered box
    """
    jittered_box_list = []
    device = boxes.device
    for box in boxes:
        scale_jitter_factor = 0.25
        center_jitter_factor = 0.25
        jittered_size = box[2:4] * torch.exp(torch.randn(2, device=device) * scale_jitter_factor)
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(center_jitter_factor, device=device).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2, device=device) - 0.5)
        jittered_box = torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0).unsqueeze(0)
        jittered_box_list.append(jittered_box)
    jittered_boxes = torch.cat(jittered_box_list, dim=0)
    return jittered_boxes

def get_jittered_box_1(box):
    """ Jitter the input box
    args:
        box - input bounding box
    returns:
        torch.Tensor - jittered box
    """
    device = box.device
    scale_jitter_factor = 0.25
    center_jitter_factor = 0.5
    jittered_size = box[2:4] * torch.exp(torch.randn(2, device=device) * scale_jitter_factor)
    max_offset = (jittered_size.prod().sqrt() * torch.tensor(center_jitter_factor, device=device).float())
    jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2, device=device) - 0.5)
    jittered_box = torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)
    return jittered_box