""" Dataset loader for the Charades-STA dataset """
import torch
from torch import nn
import torch.utils.data as data
from datasets.transforms import feature_temporal_sampling
import os
import json
import csv
import torchtext
import torchvision
import numpy as np
from core.utils import iou, ioa
import h5py
import torch.nn.functional as F
import random
from nltk.tokenize import word_tokenize
# random.seed(0)
import nltk
nltk.download('punkt')

class DatasetBase(data.Dataset):
    def __init__(self, cfg, split):
        super(DatasetBase, self).__init__()
        self.cfg = cfg
        self.split = split
        self.annotations = None
        getattr(self, "get_{}_annotations".format(self.cfg.NAME))()

    def __len__(self):
        return len(self.annotations)

    def get_pos_embedding(self, duration, num_clips, num_anchors, time_unit):
        s_times = torch.arange(0, num_clips).float() * time_unit

        s_times = s_times[:, None].expand(-1, num_anchors)
        lengths = torch.arange(1, num_anchors + 1).float() * time_unit
        lengths = lengths[None, :].expand(num_clips, -1)
        e_times = s_times + lengths
        pos_embedding = torch.stack([s_times, e_times], dim=2) / duration
        return pos_embedding

    def get_iou_map(self, gt_times, duration, num_clips, num_anchors):
        gt_s_time, gt_e_time = gt_times

        s_times = torch.arange(0, num_clips).float() * duration / num_clips
        s_times = s_times[:, None].expand(-1, num_anchors)
        lengths = torch.arange(1, num_anchors + 1).float() * duration / num_clips
        lengths = lengths[None, :].expand(num_clips, -1)
        e_times = s_times + lengths
        overlaps = iou(torch.stack([s_times, e_times], dim=2).view(-1, 2).tolist(),
                       torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_anchors)
        for i in range(1, min(num_anchors, num_clips)):
            overlaps[num_clips - i, i:] = 0
        # gt_s_idx = np.argmax(overlaps) // num_anchors
        # gt_e_idx = np.argmax(overlaps) % num_anchors
        return overlaps

    def get_ioa_map(self, gt_times, duration, num_clips, num_anchors):
        gt_s_time, gt_e_time = gt_times

        s_times = torch.arange(0, num_clips).float() * duration / num_clips
        s_times = s_times[:, None].expand(-1, num_anchors)
        lengths = torch.arange(1, num_anchors + 1).float() * duration / num_clips
        lengths = lengths[None, :].expand(num_clips, -1)
        e_times = s_times + lengths
        delta = self.cfg.DELTA_SIZE * duration / num_clips
        start_overlaps = ioa(torch.stack([s_times-delta / 2, s_times+delta / 2], dim=2).view(-1, 2).tolist(),
                             torch.tensor([gt_s_time-delta / 2, gt_s_time+delta / 2]).tolist()).reshape(num_clips, num_anchors)
        end_overlaps = ioa(torch.stack([e_times-delta / 2, e_times+delta / 2], dim=2).view(-1, 2).tolist(),
                             torch.tensor([gt_e_time-delta / 2, gt_e_time+delta / 2]).tolist()).reshape(num_clips, num_anchors)
        # start_overlaps = ioa(torch.stack([s_times, s_times+delta], dim=2).view(-1, 2).tolist(),
        #                      torch.tensor([gt_s_time, gt_s_time+delta]).tolist()).reshape(num_clips, num_anchors)
        # end_overlaps = ioa(torch.stack([e_times, e_times+delta], dim=2).view(-1, 2).tolist(),
        #                      torch.tensor([gt_e_time, gt_e_time+delta]).tolist()).reshape(num_clips, num_anchors)

        for i in range(1, min(num_anchors, num_clips)):
            start_overlaps[num_clips - i, i:] = 0
            end_overlaps[num_clips - i, i:] = 0

        return start_overlaps, end_overlaps

    def get_ioa_line(self, gt_times, duration, num_clips):
        gt_s_time, gt_e_time = gt_times

        s_times = torch.arange(0, num_clips).float() * duration / num_clips
        e_times = torch.arange(1, num_clips+1).float() * duration / num_clips
        overlap = ioa(torch.stack([s_times, e_times], dim=1).view(-1, 2).tolist(),
                      torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips)
        return overlap

    def get_video_frames(self, video_id):
        if self.cfg.NAME == "charades":
            video_path = os.path.join(self.cfg.DATA_DIR, 'Charades_v1_480', video_id + '.mp4')
        else:
            raise NotImplementedError
        pts, fps = torchvision.io.read_video_timestamps(video_path, pts_unit='sec')
        vframes = torchvision.io.read_video(video_path, start_pts=pts[0], end_pts=pts[-1], pts_unit='sec')[0]
        return vframes

    def get_video_features(self, video_id):
        file_path = os.path.join(self.cfg.DATA_DIR, '{}.hdf5'.format(self.cfg.VIS_INPUT_TYPE))
        with h5py.File(file_path, 'r') as hdf5_file:
            features = torch.from_numpy(hdf5_file[video_id][:]).float()
        if self.cfg.NORMALIZE:
            features = F.normalize(features,dim=1)
        mask = torch.ones(features.shape[0], 1)
        return features, mask

    def __getitem__(self, index):
        if self.cfg.SLIDING_WINDOW:
            return self.get_sliding_window_item(index)
        else:
            return self.get_sampling_item(index)

    def get_sliding_window_item(self, index):
        raise NotImplementedError

    def get_sampling_item(self, index):
        raise NotImplementedError

class MomentLocalizationDataset(DatasetBase):
    def __init__(self, cfg, split):
        super(MomentLocalizationDataset, self).__init__(cfg, split)
        self.annotations = sorted(self.annotations, key=lambda anno: anno['duration'], reverse=True)

        vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"](cache=os.path.join(self.cfg.DATA_DIR, '.vector_cache'))
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
        word_embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.vocab = vocab
        self.word_embedding = word_embedding

    def get_sentence_features(self, description):
        if self.cfg.TXT_INPUT_TYPE == 'glove':
            word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in description.split()],
                                     dtype=torch.long)
            word_vectors = self.word_embedding(word_idxs)
        else:
            raise NotImplementedError
        return word_vectors, torch.ones(word_vectors.shape[0], 1)

    def get_charades_annotations(self):
        durations = {}
        with open(os.path.join(self.cfg.DATA_DIR, 'Charades_v1_{}.csv'.format(self.split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                durations[row['id']] = float(row['length'])

        anno_file = open(os.path.join(self.cfg.DATA_DIR, "charades_sta_{}.txt".format(self.split)), 'r')
        annotations = []
        for line in anno_file:
            anno, sent = line.split("##")
            sent = sent.split('.\n')[0]
            vid, s_time, e_time = anno.split(" ")
            duration = durations[vid]
            s_time = float(s_time)
            e_time = min(float(e_time), duration)
            if s_time < e_time:
                annotations.append(
                    {'video': vid, 'times': [s_time, e_time], 'description': sent,
                     'duration': duration})
        anno_file.close()
        self.annotations = annotations

    def get_activitynet_annotations(self):
        with open(os.path.join(self.cfg.DATA_DIR, '{}.json'.format(self.split)), 'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        missing_videos = []#'v_0dkIbKXXFzI'
        for vid, video_anno in annotations.items():
            if vid in missing_videos:
                continue
            duration = video_anno['duration']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append(
                        {
                            'video': vid,
                            'duration': duration,
                            'times': [max(timestamp[0], 0), min(timestamp[1], duration)],
                            'description': ' '.join(word_tokenize(sentence)),
                        }
                    )
        self.annotations = anno_pairs

    def get_tacos_annotations(self):
        with open(os.path.join(self.cfg.DATA_DIR, '{}.json'.format(self.split)),'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        for vid, video_anno in annotations.items():
            duration = video_anno['num_frames']/video_anno['fps']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append(
                        {
                            'video': os.path.splitext(vid)[0],
                            'duration': duration,
                            'times':[max(timestamp[0]/video_anno['fps'],0),min(timestamp[1]/video_anno['fps'],duration)],
                            'description':' '.join(word_tokenize(sentence)),
                        }
                    )
        self.annotations = anno_pairs

    def get_tvretrieval_annotations(self):
        raise NotImplementedError

    def get_sliding_window_item(self, index):
        # index = 752#2548#3951#837#3951
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        description = self.annotations[index]['description']
        duration = self.annotations[index]['duration']
        time_unit = self.cfg.TIME_UNIT

        word_vectors, txt_mask = self.get_sentence_features(description)
        video_features, vis_mask = self.get_video_features(video_id)

        num_clips = video_features.shape[0]
        # assert abs(num_clips*time_unit - duration) < 2*time_unit

        if "train" in self.split:
            rand_s_idx = random.randrange(num_clips - self.cfg.INPUT_NUM_CLIPS) if num_clips > self.cfg.INPUT_NUM_CLIPS else 0
            rand_e_idx = min(rand_s_idx + self.cfg.INPUT_NUM_CLIPS, num_clips)
            video_features = video_features[rand_s_idx:rand_e_idx]
            vis_mask = vis_mask[rand_s_idx:rand_e_idx]
            if self.cfg.INPUT_NUM_CLIPS > num_clips:
                video_features = F.pad(video_features, [0, 0, 0, self.cfg.INPUT_NUM_CLIPS - num_clips])
                vis_mask = F.pad(vis_mask, [0, 0, 0, self.cfg.INPUT_NUM_CLIPS - num_clips])
                if isinstance(self.cfg.OUTPUT_NUM_CLIPS, int) and isinstance(self.cfg.NUM_ANCHORS, int):
                    downsample_rate = self.cfg.INPUT_NUM_CLIPS // self.cfg.OUTPUT_NUM_CLIPS
                    scale_num_clips = max(num_clips // downsample_rate, 1)
                    scale_duration = time_unit * downsample_rate * scale_num_clips
                    overlaps = self.get_iou_map([gt_s_time, gt_e_time], scale_duration, scale_num_clips, self.cfg.NUM_ANCHORS)
                    overlaps = torch.from_numpy(overlaps)
                    overlaps = F.pad(overlaps, [0, 0, 0, self.cfg.OUTPUT_NUM_CLIPS - scale_num_clips])
                    pos_embedding = self.get_pos_embedding(duration, scale_num_clips, self.cfg.NUM_ANCHORS, time_unit*downsample_rate)
                    item = {'map_gt': overlaps,
                            'pos_emb': pos_embedding}

                elif isinstance(self.cfg.OUTPUT_NUM_CLIPS, list) and isinstance(self.cfg.NUM_ANCHORS, list):
                    multi_overlaps, multi_pos_emb = [], []
                    for num_out_clips, num_anchors in zip(self.cfg.OUTPUT_NUM_CLIPS, self.cfg.NUM_ANCHORS):
                        downsample_rate = self.cfg.INPUT_NUM_CLIPS // num_out_clips
                        scale_num_clips = max(num_clips // downsample_rate, 1)
                        scale_duration = time_unit*downsample_rate*scale_num_clips
                        overlaps = self.get_iou_map([gt_s_time, gt_e_time], scale_duration, scale_num_clips, num_anchors)
                        overlaps = torch.from_numpy(overlaps)
                        overlaps = F.pad(overlaps, [0, 0, 0, num_out_clips - scale_num_clips])
                        multi_overlaps.append(overlaps)
                        pos_embedding = self.get_pos_embedding(duration, num_out_clips, num_anchors,
                                                               time_unit * downsample_rate)
                        multi_pos_emb.append(pos_embedding)

                    item = {'map_gt': multi_overlaps,
                            'pos_emb': multi_pos_emb}
                else:
                    raise NotImplementedError

            else:
                gt_s_time = gt_s_time - rand_s_idx*time_unit
                gt_e_time = gt_e_time - rand_s_idx*time_unit
                if isinstance(self.cfg.OUTPUT_NUM_CLIPS, int) and isinstance(self.cfg.NUM_ANCHORS, int):
                    downsample_rate = self.cfg.INPUT_NUM_CLIPS // self.cfg.OUTPUT_NUM_CLIPS
                    scale_num_clips = max(num_clips // downsample_rate, 1)
                    scale_duration = time_unit * downsample_rate * scale_num_clips
                    overlaps = self.get_iou_map([gt_s_time, gt_e_time], scale_duration, self.cfg.OUTPUT_NUM_CLIPS, self.cfg.NUM_ANCHORS)
                    overlaps = torch.from_numpy(overlaps)
                    pos_embedding = self.get_pos_embedding(duration, self.cfg.OUTPUT_NUM_CLIPS, self.cfg.NUM_ANCHORS, time_unit*downsample_rate)
                    item = {'map_gt': overlaps,
                            'pos_emb': pos_embedding}
                elif isinstance(self.cfg.OUTPUT_NUM_CLIPS, list) and isinstance(self.cfg.NUM_ANCHORS, list):
                    multi_overlaps, multi_pos_emb = [], []
                    for num_out_clips, num_anchors in zip(self.cfg.OUTPUT_NUM_CLIPS, self.cfg.NUM_ANCHORS):
                        downsample_rate = self.cfg.INPUT_NUM_CLIPS // num_out_clips
                        scale_duration = time_unit * downsample_rate * num_out_clips
                        overlaps = self.get_iou_map([gt_s_time, gt_e_time], scale_duration, num_out_clips, num_anchors)
                        overlaps = torch.from_numpy(overlaps)
                        multi_overlaps.append(overlaps)

                        pos_embedding = self.get_pos_embedding(duration, num_out_clips, num_anchors,
                                                               time_unit * downsample_rate) + rand_s_idx * time_unit / duration
                        multi_pos_emb.append(pos_embedding)
                    item = {'map_gt': multi_overlaps,
                            'pos_emb': multi_pos_emb}
                else:
                    raise NotImplementedError

        else:

            if isinstance(self.cfg.OUTPUT_NUM_CLIPS, int) and isinstance(self.cfg.NUM_ANCHORS, int):
                downsample_rate = self.cfg.INPUT_NUM_CLIPS // self.cfg.OUTPUT_NUM_CLIPS
                scale_num_clips = max(num_clips // downsample_rate, 1)
                scale_duration = time_unit * downsample_rate * scale_num_clips
                overlaps = self.get_iou_map([gt_s_time, gt_e_time], scale_duration, scale_num_clips, self.cfg.NUM_ANCHORS)
                overlaps = torch.from_numpy(overlaps)
                pos_embedding = self.get_pos_embedding(duration, self.cfg.OUTPUT_NUM_CLIPS, self.cfg.NUM_ANCHORS,
                                                       time_unit * downsample_rate)
                if self.cfg.INPUT_NUM_CLIPS > num_clips:
                    video_features = F.pad(video_features, [0, 0, 0, self.cfg.INPUT_NUM_CLIPS - num_clips])
                    vis_mask = F.pad(vis_mask, [0, 0, 0, self.cfg.INPUT_NUM_CLIPS - num_clips])
                    overlaps = F.pad(overlaps, [0, 0, 0, self.cfg.OUTPUT_NUM_CLIPS - scale_num_clips])
                item = {'map_gt': overlaps,
                        'pos_emb': pos_embedding}

            elif isinstance(self.cfg.OUTPUT_NUM_CLIPS, list) and isinstance(self.cfg.NUM_ANCHORS, list):
                multi_overlaps, multi_pos_emb = [], []
                for num_out_clips, num_anchors in zip(self.cfg.OUTPUT_NUM_CLIPS, self.cfg.NUM_ANCHORS):
                    downsample_rate = self.cfg.INPUT_NUM_CLIPS // num_out_clips
                    scale_num_clips = max(num_clips // downsample_rate, 1)
                    scale_duration = time_unit * downsample_rate * scale_num_clips
                    overlaps = self.get_iou_map([gt_s_time, gt_e_time], scale_duration, scale_num_clips, num_anchors)
                    overlaps = torch.from_numpy(overlaps)
                    pos_embedding = self.get_pos_embedding(duration, max(scale_num_clips, num_out_clips), num_anchors,
                                                           time_unit * downsample_rate)
                    multi_pos_emb.append(pos_embedding)
                    if self.cfg.INPUT_NUM_CLIPS > num_clips:
                        overlaps = F.pad(overlaps, [0, 0, 0, num_out_clips - scale_num_clips])
                    multi_overlaps.append(overlaps)
                if self.cfg.INPUT_NUM_CLIPS > num_clips:
                    video_features = F.pad(video_features, [0, 0, 0, self.cfg.INPUT_NUM_CLIPS - num_clips])
                    vis_mask = F.pad(vis_mask, [0, 0, 0, self.cfg.INPUT_NUM_CLIPS - num_clips])
                item = {'map_gt': multi_overlaps,
                        'pos_emb': multi_pos_emb}

        item.update({
            'video_id': video_id,
            'video_features': video_features,
            'vis_mask': vis_mask,
            'description': description,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'txt_mask': txt_mask,
        })

        return item


    def get_sampling_item(self, index):
        # index = 13740#[8973, 13740]
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        description = self.annotations[index]['description']
        duration = self.annotations[index]['duration']

        word_vectors, txt_mask = self.get_sentence_features(description)
        video_features, vis_mask = self.get_video_features(video_id)
        video_features = feature_temporal_sampling(self.cfg.INPUT_NUM_CLIPS, video_features)
        vis_mask = feature_temporal_sampling(self.cfg.INPUT_NUM_CLIPS, vis_mask)
        item = {
            'video_id': video_id,
            'description': description,
            'video_features': video_features,
            'vis_mask': vis_mask,
            'anno_idx': index,
            'word_vectors': word_vectors,
            'txt_mask': txt_mask,
            'duration': duration
        }

        if isinstance(self.cfg.OUTPUT_NUM_CLIPS, list) and isinstance(self.cfg.NUM_ANCHORS, list):
            prop_overlaps_list, start_overlaps_list, end_overlaps_list, seg_overlaps_list = [], [], [], []
            for num_clips, num_anchors in zip(self.cfg.OUTPUT_NUM_CLIPS, self.cfg.NUM_ANCHORS):
                prop_overlap = self.get_iou_map([gt_s_time, gt_e_time], duration, num_clips, num_anchors)
                start_overlap, end_overlap = self.get_ioa_map([gt_s_time, gt_e_time], duration, num_clips, num_anchors)
                prop_overlap = torch.from_numpy(prop_overlap)
                start_overlap = torch.from_numpy(start_overlap)
                end_overlap = torch.from_numpy(end_overlap)
                seg_overlap = self.get_ioa_line([gt_s_time, gt_e_time], duration, num_clips)
                seg_overlap = torch.from_numpy(seg_overlap)
                prop_overlaps_list.append(prop_overlap)
                start_overlaps_list.append(start_overlap)
                end_overlaps_list.append(end_overlap)
                seg_overlaps_list.append(seg_overlap)
            item.update({'map_gt': prop_overlaps_list})
            item.update({'start_gt': start_overlaps_list})
            item.update({'end_gt': end_overlaps_list})
            item.update({'seg_gt': seg_overlaps_list})
        elif isinstance(self.cfg.OUTPUT_NUM_CLIPS, int) and isinstance(self.cfg.NUM_ANCHORS, int):
            prop_overlap = self.get_iou_map([gt_s_time, gt_e_time], duration, self.cfg.OUTPUT_NUM_CLIPS, self.cfg.NUM_ANCHORS)
            prop_overlap = torch.from_numpy(prop_overlap)
            seg_overlap = self.get_ioa_line([gt_s_time, gt_e_time], duration, self.cfg.OUTPUT_NUM_CLIPS)
            seg_overlap = torch.from_numpy(seg_overlap)
            item.update({'map_gt': prop_overlap})
            item.update({'seg_gt': seg_overlap})
        else:
            raise NotImplementedError

        return item