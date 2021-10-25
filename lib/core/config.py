from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

cfg = edict()

cfg.WORKERS = 16
cfg.LOG_DIR = ''
cfg.MODEL_DIR = ''
cfg.RESULT_DIR = ''

# CUDNN related params
cfg.CUDNN = edict()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

# grounding model related params
cfg.MODEL = edict()
cfg.MODEL.NAME = ''
cfg.MODEL.PARAMS = None
cfg.MODEL.CHECKPOINT = '' # The checkpoint for the best performance
cfg.MODEL.CLIP_MODULE = edict()
cfg.MODEL.CLIP_MODULE.NAME = ''
cfg.MODEL.CLIP_MODULE.PARAMS = None
cfg.MODEL.PROP_MODULE = edict()
cfg.MODEL.PROP_MODULE.NAME = ''
cfg.MODEL.PROP_MODULE.PARAMS = None
cfg.MODEL.FUSION_MODULE = edict()
cfg.MODEL.FUSION_MODULE.NAME = ''
cfg.MODEL.FUSION_MODULE.PARAMS = None
cfg.MODEL.MAP_MODULE = edict()
cfg.MODEL.MAP_MODULE.NAME = ''
cfg.MODEL.MAP_MODULE.PARAMS = None
cfg.MODEL.PRED_MODULE = edict()
cfg.MODEL.PRED_MODULE.NAME = ''
cfg.MODEL.PRED_MODULE.PARAMS = None

# DATASET related params
cfg.DATASET = edict()
cfg.DATASET.DATA_DIR = ''
cfg.DATASET.NAME = ''
cfg.DATASET.VIS_INPUT_TYPE = ''
cfg.DATASET.TXT_INPUT_TYPE = ''
cfg.DATASET.OUTPUT_TYPE = ''
cfg.DATASET.ALIGNMENT = ''
cfg.DATASET.NO_VAL = False
cfg.DATASET.NO_TEST = False
cfg.DATASET.TIME_UNIT = None
cfg.DATASET.NUM_FRAMES = 256
cfg.DATASET.INPUT_NUM_CLIPS = 256
cfg.DATASET.OUTPUT_NUM_CLIPS = 16
cfg.DATASET.INPUT_CLIP_SIZE = 16
cfg.DATASET.OUTPUT_CLIP_SIZE = 1
cfg.DATASET.NUM_ANCHORS = 16
cfg.DATASET.SPLIT = ''
cfg.DATASET.NORMALIZE = True
cfg.DATASET.SLIDING_WINDOW = False
cfg.DATASET.DELTA_SIZE = 1
cfg.DATASET.QUERY_NUM_CLIPS = 8
cfg.DATASET.FPS = 25
cfg.DATASET.VIDEO_WINDOW_SIZE = 64

# OPTIM
cfg.OPTIM = edict()
cfg.OPTIM.NAME = ''
cfg.OPTIM.PARAMS = edict()
cfg.OPTIM.SCHEDULER = edict()
cfg.OPTIM.SCHEDULER.FACTOR = 0.5
cfg.OPTIM.SCHEDULER.PATIENCE = 500
# train
cfg.TRAIN = edict()
cfg.TRAIN.MAX_EPOCH = 20
cfg.TRAIN.BATCH_SIZE = 4
cfg.TRAIN.SHUFFLE = True
cfg.TRAIN.CONTINUE = False

cfg.LOSS = edict()
cfg.LOSS.NAME = 'bce_loss'
cfg.LOSS.PARAMS = None

# test
cfg.TEST = edict()
cfg.TEST.RECALL = []
cfg.TEST.TIOU = []
cfg.TEST.NMS_THRESH = 0.4
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.TOP_K = 10
cfg.TEST.EVAL_TRAIN = False

def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if 'PARAMS' in k:
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k],v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))

def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    _update_dict(cfg[k], v)
                else:
                    cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
