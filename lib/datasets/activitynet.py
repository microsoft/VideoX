""" Dataset loader for the ActivityNet Captions dataset """
import os
import json

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from . import average_to_fixed_length
from core.eval import iou
from core.config import config

class ActivityNet(data.Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"]()
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super(ActivityNet, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        # val_1.json is renamed as val.json, val_2.json is renamed as test.json
        with open(os.path.join(self.data_dir, '{}.json'.format(split)),'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        for vid, video_anno in annotations.items():
            duration = video_anno['duration']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append(
                        {
                            'video': vid,
                            'duration': duration,
                            'times':[max(timestamp[0],0),min(timestamp[1],duration)],
                            'description':sentence,
                        }
                    )
        self.annotations = anno_pairs

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        sentence = self.annotations[index]['description']
        duration = self.annotations[index]['duration']

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()], dtype=torch.long)
        word_vectors = self.word_embedding(word_idxs)

        visual_input, visual_mask = self.get_video_features(video_id)

        # max_idx = visual_input.shape[0]
        # gt_s_idx = max(time_to_index(gt_s_time),0)
        # gt_e_idx = min(time_to_index(gt_e_time),max_idx-1)
        #
        # map_gt = torch.zeros(1,max_idx,max_idx)
        #
        # max_time = (gt_e_time-gt_s_time)/config.LOSS.MIN_IOU
        # min_time = (gt_e_time-gt_s_time)*config.LOSS.MIN_IOU
        # s_idx_s = max(time_to_index(gt_e_time - max_time), 0)
        # s_idx_e = time_to_index(gt_s_time + min_time)
        # e_idx_s = time_to_index(gt_e_time - min_time)
        # e_idx_e = min(time_to_index(gt_s_time + max_time), max_idx-1)
        #
        # s_idxs = torch.arange(s_idx_s, s_idx_e+1)
        # e_idxs = torch.arange(e_idx_s, e_idx_e+1)
        # s_times = index_to_time(s_idxs.float())
        # e_times = index_to_time(e_idxs.float()+1)
        #
        #
        # overlaps = iou(torch.stack([s_times[:,None].expand(-1,e_idx_e+1-e_idx_s),
        #                             e_times[None,:].expand(s_idx_e+1-s_idx_s,-1)],dim=2).view(-1,2).tolist(),
        #                torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(s_idxs.shape[0],e_idxs.shape[0])
        # for s_idx in range(s_idx_s, s_idx_e+1):
        #     for e_idx in range(e_idx_s, e_idx_e+1):
        #         map_gt[0,s_idx,e_idx] = overlaps[s_idx-s_idx_s,e_idx-e_idx_s]

        # Time scaled to same size
        if config.DATASET.NUM_SAMPLE_CLIPS > 0:
            # visual_input = sample_to_fixed_length(visual_input, random_sampling=True)
            visual_input = average_to_fixed_length(visual_input)
            num_clips = config.DATASET.NUM_SAMPLE_CLIPS//config.DATASET.TARGET_STRIDE
            s_times = torch.arange(0,num_clips).float()*duration/num_clips
            e_times = torch.arange(1,num_clips+1).float()*duration/num_clips
            overlaps = iou(torch.stack([s_times[:,None].expand(-1,num_clips),
                                        e_times[None,:].expand(num_clips,-1)],dim=2).view(-1,2).tolist(),
                           torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips,num_clips)

        # Time unscaled NEED FIXED WINDOW SIZE
        else:
            num_clips = visual_input.shape[0]//config.DATASET.TARGET_STRIDE
            raise NotImplementedError
            # torch.arange(0,)

        item = {
            'visual_input': visual_input,
            'vis_mask': visual_mask,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'duration': duration,
            'txt_mask': torch.ones(word_vectors.shape[0], 1),
            'map_gt': torch.from_numpy(overlaps),
        }

        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
        with h5py.File(os.path.join(self.data_dir, 'sub_activitynet_v1-3.c3d.hdf5'), 'r') as f:
            features = torch.from_numpy(f[vid]['c3d_features'][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features,dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask