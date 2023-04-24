# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

class CosineEMA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_epoch = 0

    def adjust(self, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        self.last_epoch = epoch
        cfg = self.cfg

        ema_decay = cfg.TRAIN.EMA_SCHEDULER.MAX - \
             (cfg.TRAIN.EMA_SCHEDULER.MAX - cfg.TRAIN.EMA_SCHEDULER.MIN) * 0.5 * \
            (1. + math.cos(math.pi * epoch / cfg.TRAIN.EPOCH))

        return ema_decay

class ConstantEMA:
    def __init__(self, cfg):
        self.cfg = cfg

    def adjust(self, epoch):
        ema_decay = self.cfg.TRAIN.EMA_DECAY
        return ema_decay