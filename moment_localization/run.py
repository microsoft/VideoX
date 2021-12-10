from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from datasets.dataset import MomentLocalizationDataset
from core.config import cfg, update_config
import torch.nn.functional as F
from core.utils import AverageMeter, create_logger
import eval
import models
import models.loss as loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--seed', help='seed', default=0, type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    parser.add_argument('--mode', default='train', help='run test epoch only')
    parser.add_argument('--split', help='test split', type=str)
    parser.add_argument('--no_save', default=False, action="store_true", help='don\'t save checkpoint')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers is not None:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATASET.DATA_DIR = os.path.join(args.dataDir, config.DATASET.DATA_DIR)
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.tag:
        config.TAG = args.tag

def collate_fn(batch):
    batch_word_vectors = [b['word_vectors'] for b in batch]
    batch_txt_mask = [b['txt_mask'] for b in batch]
    batch_vis_mask = [b['vis_mask'] for b in batch]
    batch_map_gt = [b['map_gt'] for b in batch]
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_video_ids = [b['video_id'] for b in batch]
    batch_video_features = [b['video_features'] for b in batch]
    batch_descriptions = [b['description'] for b in batch]

    batch_data = {
        'batch_video_ids': batch_video_ids,
        'batch_anno_idxs': batch_anno_idxs,
        'batch_descriptions': batch_descriptions,
        'batch_word_vectors': nn.utils.rnn.pad_sequence(batch_word_vectors, batch_first=True),
        'batch_txt_mask': nn.utils.rnn.pad_sequence(batch_txt_mask, batch_first=True),
        'batch_map_gt': [nn.utils.rnn.pad_sequence(map_gt, batch_first=True).float()[:, None] for map_gt in zip(*batch_map_gt)],
        'batch_video_features': nn.utils.rnn.pad_sequence(batch_video_features, batch_first=True).float().transpose(1, 2),
        'batch_vis_mask': nn.utils.rnn.pad_sequence(batch_vis_mask, batch_first=True).float().transpose(1, 2),
    }


    if cfg.DATASET.SLIDING_WINDOW:
        batch_pos_emb = [b['pos_emb'] for b in batch]
        batch_data.update({
            'batch_pos_emb': [nn.utils.rnn.pad_sequence(pos_emb, batch_first=True).float().permute(0,3,1,2) for pos_emb in zip(*batch_pos_emb)]
        })
    else:
        batch_data.update({
            'batch_duration': [b['duration'] for b in batch]
        })

    return batch_data

def recover_to_single_map(joint_probs):
    batch_size, _, map_size, _ = joint_probs[0].shape
    score_map = torch.zeros(batch_size, 1, map_size, map_size).cuda()
    for prob in joint_probs:
        scale_num_clips, scale_num_anchors = prob.shape[2:]
        dilation = map_size//scale_num_clips
        for i in range(scale_num_anchors):
            score_map[...,:map_size//dilation*dilation:dilation,(i+1)*dilation-1] = torch.max(
                score_map[...,:map_size//dilation*dilation:dilation,(i+1)*dilation-1].clone(), prob[...,i])
    return score_map

def upsample_to_single_map(joint_probs):
    batch_size, _, map_size, _ = joint_probs[0].shape
    score_map = torch.zeros(batch_size, 1, map_size, map_size).cuda()
    for i, prob in enumerate(joint_probs):
        dilation = 2**(i)
        num_clips, num_anchors = prob.shape[-2:]
        score_map[...,:dilation*num_clips, :dilation*num_anchors] = torch.max(
            F.interpolate(prob, scale_factor=dilation, mode='bilinear', align_corners=True),
            score_map[..., :dilation * num_clips, :dilation * num_anchors]
        )
    return score_map

def network(sample, model, optimizer=None, return_map=False):
    textual_input = sample['batch_word_vectors']
    textual_mask = sample['batch_txt_mask']
    visual_mask = sample['batch_vis_mask']
    visual_input = sample['batch_video_features']
    map_gts = sample['batch_map_gt']

    predictions, map_masks = model(textual_input, textual_mask, visual_input, visual_mask)

    loss_value = 0
    for prediction, map_mask, map_gt in zip(predictions, map_masks, map_gts):
        scale_loss = getattr(loss, cfg.LOSS.NAME)(prediction, map_mask, map_gt.cuda(), cfg.LOSS.PARAMS)
        loss_value += scale_loss
    joint_prob = recover_to_single_map(predictions)
    mask = recover_to_single_map(map_masks)

    if torch.sum(mask[0] > 0).item() == 0:
        print(sample['batch_anno_idxs'])
    assert torch.sum(mask[0] > 0).item() > 0

    if cfg.DATASET.SLIDING_WINDOW:
        time_unit = cfg.DATASET.TIME_UNIT*cfg.DATASET.INPUT_NUM_CLIPS/cfg.DATASET.OUTPUT_NUM_CLIPS[0]
        sorted_times = get_sw_proposal_results(joint_prob.detach().cpu(), mask, time_unit)
    else:
        sorted_times = get_proposal_results(joint_prob.detach().cpu(), mask, sample['batch_duration'])

    if model.training:
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    if return_map:
        return loss_value, sorted_times, joint_prob.detach().cpu()
    else:
        return loss_value, sorted_times

from eval import nms
def get_proposal_results(scores, mask, durations):
    # assume all valid scores are larger than one
    out_sorted_times = []
    batch_size, _, num_clips, num_anchors = scores.shape
    scores, indexes = torch.topk(scores.view(batch_size, -1), torch.sum(mask[0] > 0).item(), dim=1)
    t_starts = (indexes // num_anchors) .float()/num_clips*torch.tensor(durations).view(batch_size,1)
    t_ends = t_starts + (indexes % num_anchors + 1).float()/num_clips*torch.tensor(durations).view(batch_size,1)

    for t_start, t_end in zip(t_starts, t_ends):
        t_start, t_end = t_start[t_start < t_end], t_end[t_start<t_end]
        dets = nms(torch.stack([t_start,t_end],dim=1).tolist(), thresh=cfg.TEST.NMS_THRESH, top_k=max(cfg.TEST.RECALL))
        out_sorted_times.append(dets)
    return out_sorted_times

def get_sw_proposal_results(scores, mask, time_unit):
    # assume all valid scores are larger than one
    out_sorted_times = []
    batch_size, _, num_clips, num_anchors = scores.shape
    scores, indexes = torch.topk(scores.view(batch_size, -1), torch.sum(mask[0] > 0).item(), dim=1)
    t_starts = (indexes // num_anchors) .float()*time_unit
    t_ends = t_starts + (indexes % num_anchors + 1).float()*time_unit

    for t_start, t_end in zip(t_starts, t_ends):
        t_start, t_end = t_start[t_start < t_end], t_end[t_start<t_end]
        dets = nms(torch.stack([t_start,t_end],dim=1).tolist(), thresh=cfg.TEST.NMS_THRESH, top_k=max(cfg.TEST.RECALL))
        out_sorted_times.append(dets)
    return out_sorted_times

def train_epoch(train_loader, model, optimizer, verbose=False):
    model.train()

    loss_meter = AverageMeter()
    sorted_segments_dict = {}
    if verbose:
        pbar = tqdm(total=len(train_loader), dynamic_ncols=True)

    for cur_iter, sample in enumerate(train_loader):
        loss_value, sorted_times = network(sample, model, optimizer)
        loss_meter.update(loss_value.item(), 1)
        sorted_segments_dict.update({idx: timestamp for idx, timestamp in zip(sample['batch_anno_idxs'], sorted_times)})
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    annotations = train_loader.dataset.annotations
    sorted_segments = [sorted_segments_dict[key] for key in sorted(sorted_segments_dict.keys())]
    result = eval.evaluate(sorted_segments, annotations)

    return loss_meter.avg, result

@torch.no_grad()
def test_epoch(test_loader, model, verbose=False, save_results=False):
    model.eval()
    loss_meter = AverageMeter()
    sorted_segments_dict = {}
    saved_dict = {}
    if verbose:
        pbar = tqdm(total=len(test_loader), dynamic_ncols=True)
    for cur_iter, sample in enumerate(test_loader):
        loss_value, sorted_times, score_maps = network(sample, model, return_map=True)
        loss_meter.update(loss_value.item(), 1)
        sorted_segments_dict.update({idx: timestamp for idx, timestamp in zip(sample['batch_anno_idxs'],sorted_times)})
        saved_dict.update({idx: {'vid': vid, 'timestamps': timestamp, 'description': description}
                           for idx, vid, timestamp, description in zip(sample['batch_anno_idxs'],
                                                                       sample['batch_video_ids'],
                                                                       sorted_times,
                                                                       sample['batch_descriptions'])})
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()
    annotations = test_loader.dataset.annotations
    sorted_segments = [sorted_segments_dict[key] for key in sorted(sorted_segments_dict.keys())]
    saved_dict = [saved_dict[key] for key in sorted(saved_dict.keys())]
    if save_results:
        if not os.path.exists('results/{}'.format(cfg.DATASET.NAME)):
            os.makedirs('results/{}'.format(cfg.DATASET.NAME))
        torch.save(saved_dict, 'results/{}/{}-{}.pkl'.format(cfg.DATASET.NAME, os.path.basename(args.cfg).split('.yaml')[0], test_loader.dataset.split))
    result = eval.evaluate(sorted_segments, annotations)
    return loss_meter.avg, result


def train(cfg, verbose):

    logger, final_output_dir = create_logger(cfg, args.cfg, cfg.TAG)
    logger.info('\n'+pprint.pformat(args))
    logger.info('\n' + pprint.pformat(cfg))

    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    init_epoch = 0
    model = getattr(models, cfg.MODEL.NAME)(cfg.MODEL)
    if cfg.MODEL.CHECKPOINT and cfg.TRAIN.CONTINUE:
        init_epoch = int(os.path.basename(cfg.MODEL.CHECKPOINT)[5:9])+1
        model_checkpoint = torch.load(cfg.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
        print(f"loading checkpoint: {cfg.MODEL.CHECKPOINT}")
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    if cfg.OPTIM.NAME == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIM.PARAMS.LR)
    elif cfg.OPTIM.NAME == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.OPTIM.PARAMS.LR)
    elif cfg.OPTIM.NAME == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=cfg.OPTIM.PARAMS.LR)
    else:
        raise NotImplementedError

    train_dataset = MomentLocalizationDataset(cfg.DATASET, 'train')
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=cfg.TRAIN.SHUFFLE,
                              num_workers=cfg.WORKERS,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)

    if not cfg.DATASET.NO_VAL:
        val_dataset = MomentLocalizationDataset(cfg.DATASET, 'val')
        val_loader = DataLoader(val_dataset,
                                batch_size=cfg.TEST.BATCH_SIZE,
                                shuffle=False,
                                num_workers=cfg.WORKERS,
                                pin_memory=True,
                                collate_fn=collate_fn)

    test_dataset = MomentLocalizationDataset(cfg.DATASET, 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.TEST.BATCH_SIZE,
                             shuffle=False,
                             num_workers=cfg.WORKERS,
                             pin_memory=True,
                             collate_fn=collate_fn)

    for cur_epoch in range(init_epoch, cfg.TRAIN.MAX_EPOCH):
        train_avg_loss, train_result = train_epoch(train_loader, model, optimizer, verbose)
        loss_message = '\nepoch: {} train loss {:.4f}'.format(cur_epoch, train_avg_loss)
        table_message = '\n' + eval.display_results(train_result, 'performance on training set')

        if not cfg.DATASET.NO_VAL:
            val_avg_loss, val_result = test_epoch(val_loader, model, verbose)
            loss_message += ' val loss {:.4f}'.format(val_avg_loss)
            table_message += '\n' + eval.display_results(val_result, 'performance on validation set')

        test_avg_loss, test_result = test_epoch(test_loader, model, verbose)
        loss_message += ' test loss {:.4f}'.format(test_avg_loss)
        table_message += '\n' + eval.display_results(test_result, 'performance on testing set')

        message = loss_message+table_message+'\n'
        logger.info(message)

        if not args.no_save:
            saved_model_filename = os.path.join(cfg.MODEL_DIR, '{}/{}/epoch{:04d}-{:.4f}-{:.4f}.pkl'.format(
                cfg.DATASET.NAME, os.path.basename(args.cfg).split('.yaml')[0],
                cur_epoch, test_result['ranks'][0,0], test_result['ranks'][0,1]))

            rootfolder1 = os.path.dirname(saved_model_filename)
            rootfolder2 = os.path.dirname(rootfolder1)
            rootfolder3 = os.path.dirname(rootfolder2)
            if not os.path.exists(rootfolder3):
                print('Make directory %s ...' % rootfolder3)
                os.mkdir(rootfolder3)
            if not os.path.exists(rootfolder2):
                print('Make directory %s ...' % rootfolder2)
                os.mkdir(rootfolder2)
            if not os.path.exists(rootfolder1):
                print('Make directory %s ...' % rootfolder1)
                os.mkdir(rootfolder1)

            torch.save(model.module.state_dict(), saved_model_filename)

def test(cfg, split):
    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    model = getattr(models, cfg.MODEL.NAME)(cfg.MODEL)

    if os.path.exists(cfg.MODEL.CHECKPOINT):
        model_checkpoint = torch.load(cfg.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    else:
        raise("checkpoint not exists")

    model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu" )
    model = model.to(device)


    dataset = MomentLocalizationDataset(cfg.DATASET, split)
    dataloader = DataLoader(dataset,
                        batch_size=cfg.TEST.BATCH_SIZE,
                        shuffle=False,
                        num_workers=cfg.WORKERS,
                        pin_memory=True,
                        collate_fn=collate_fn)
    avg_loss, result = test_epoch(dataloader, model, True, save_results=True)
    print(' val loss {:.4f}'.format(avg_loss))
    print(eval.display_results(result, 'performance on {} set'.format(split)))

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    reset_config(cfg, args)
    if args.mode == 'train':
        train(cfg, args.verbose)
    else:
        test(cfg, args.split)
