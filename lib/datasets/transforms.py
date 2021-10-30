import torch

def feature_temporal_sampling(num_samples, features):
    num_clips = features.shape[0]
    idxs = torch.arange(0, num_samples + 1, 1.0) / num_samples * num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_samples):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        new_visual_input.append(features[(s_idx+e_idx)//2])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input