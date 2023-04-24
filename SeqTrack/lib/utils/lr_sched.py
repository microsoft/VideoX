# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

class CosineWarmUp:

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_epoch = 0

    def adjust_learning_rate(self, optimizer, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        self.last_epoch = epoch
        cfg = self.cfg
        if epoch < cfg.TRAIN.WARMUP_EPOCHS:
            lr = cfg.TRAIN.LR * epoch / cfg.TRAIN.WARMUP_EPOCHS
        else:
            lr = cfg.TRAIN.MIN_LR + (cfg.TRAIN.LR - cfg.TRAIN.MIN_LR) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - cfg.TRAIN.WARMUP_EPOCHS) / (cfg.TRAIN.EPOCH - cfg.TRAIN.WARMUP_EPOCHS)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
