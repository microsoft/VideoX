import json
import argparse
import numpy as np
from terminaltables import AsciiTable

from core.config import cfg, update_config
from core.utils import iou

def rank(pred, gt):
    return pred.index(gt) + 1

def nms(dets, thresh=0.4, top_k=-1):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    order = np.arange(0,len(dets),1)
    dets = np.array(dets)
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    lengths = x2 - x1
    assert all(lengths > 0)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) == top_k:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep].tolist()

def evaluate(segments, data):
    tious = cfg.TEST.TIOU
    recalls = cfg.TEST.RECALL

    eval_result = [[[] for _ in recalls] for _ in tious]
    # max_recall = max(recalls)
    average_iou = []
    for seg, dat in zip(segments, data):
        # seg = nms(seg, thresh=cfg.TEST.NMS_THRESH, top_k=max_recall).tolist()
        overlap = iou(seg, [dat['times']])
        average_iou.append(np.mean(np.sort(overlap[0])[-3:]))

        for i,t in enumerate(tious):
            for j,r in enumerate(recalls):
                eval_result[i][j].append((overlap > t)[:r].any())
    eval_result = np.array(eval_result).mean(axis=-1)
    miou = np.mean(average_iou)

    result = {'ranks': eval_result, 'mIoU': miou}
    return result

def display_results(result, title=None):
    tious = cfg.TEST.TIOU
    recalls = cfg.TEST.RECALL

    display_data = [['R{}@{}'.format(i,j) for i in recalls for j in tious]+['mIoU']]
    ranks = result['ranks']*100
    miou = result['mIoU']*100
    display_data.append(['{:.02f}'.format(ranks[j][i]) for i in range(len(recalls)) for j in range(len(tious))]
                        +['{:.02f}'.format(miou)])
    table = AsciiTable(display_data, title)
    for i in range(len(tious)*len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.verbose:
        config.VERBOSE = args.verbose

if __name__ == '__main__':
    args = parse_args()
    reset_config(cfg, args)
    train_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/train_data.json', 'r'))
    val_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/val_data.json', 'r'))

    moment_frequency_dict = {}
    for d in train_data:
        times = [t for t in d['times']]
        for time in times:
            time = tuple(time)
            if time not in moment_frequency_dict.keys():
                moment_frequency_dict[time] = 0
            moment_frequency_dict[time] += 1

    prior = sorted(moment_frequency_dict, key=moment_frequency_dict.get, reverse=True)
    prior = [list(item) for item in prior]
    prediction = [prior for d in val_data]

    evaluate(prediction, val_data)