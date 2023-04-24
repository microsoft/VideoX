import argparse
import torch
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import nn
import numpy as np


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='seqtrack',
                        help='training script name')
    parser.add_argument('--config', type=str, default='seqtrack_b256', help='yaml configure file name')
    args = parser.parse_args()

    return args


def get_complexity_MHA(m:nn.MultiheadAttention, x, y):
    """(L, B, D): sequence length, batch size, dimension"""
    d_mid = m.embed_dim
    query, key, value = x[0], x[1], x[2]
    Lq, batch, d_inp = query.size()
    Lk = key.size(0)
    """compute flops"""
    total_ops = 0
    # projection of Q, K, V
    total_ops += d_inp * d_mid * Lq * batch  # query
    total_ops += d_inp * d_mid * Lk * batch * 2  # key and value
    # compute attention
    total_ops += Lq * Lk * d_mid * 2
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def evaluate(model, images_list, xz, input_start, hanning, bs):
    """Compute FLOPs, Params, and Speed"""
    custom_ops = {nn.MultiheadAttention: get_complexity_MHA}
    # encoder
    macs1, params1 = profile(model, inputs=(images_list, None, input_start, "encoder"), custom_ops=custom_ops ,verbose=True)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('encoder macs is ', macs)
    print('encoder params is ', params)
    # decoder
    macs2, params2 = profile(model, inputs=(None, xz, input_start, "decoder"), custom_ops=custom_ops, verbose=True)
    macs, params = clever_format([macs2, params2], "%.3f")
    print('decoder macs is ', macs)
    print('decoder params is ', params)
    # the whole model
    macs, params = clever_format([macs1 + macs2, params1 + params2], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    '''Speed Test'''
    T_w = 100
    T_t = 1000
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(images_list, None, input_start, "encoder")
            _ = model.inference_decoder(xz, input_start, window=hanning, num_frames=1)
        start = time.time()
        for i in range(T_t):
            _ = model(images_list, None, input_start, "encoder")
            _ = model.inference_decoder(xz, input_start, window=hanning, num_frames=1)
        end = time.time()
        avg_lat = (end - start) / (T_t * bs)
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))



def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch

if __name__ == "__main__":
    device = "cuda:1"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    bins = cfg.MODEL.BINS
    start = bins + 1
    input_start = (torch.ones([bs, 1]).to(device) * start).type(dtype=torch.int64)
    seq_list = [input_start]
    input_seqs = torch.cat(seq_list, dim=1)
    if cfg.TEST.WINDOW == True:
        hanning = torch.tensor(np.hanning(bins)).unsqueeze(0).cuda()
    else:
        hanning = None
    '''import seqtrack network module'''
    model_module = importlib.import_module('lib.models.seqtrack')
    model_constructor = model_module.build_seqtrack
    model = model_constructor(cfg)
    # get the template and search
    template = get_data(bs, z_sz)
    search = get_data(bs, x_sz)
    # transfer to device
    model = model.to(device)
    template = template.to(device)
    search = search.to(device)
    model.eval()
    # evaluate the model properties
    images_list = [template, template, search]
    xz = model.forward_encoder(images_list)
    evaluate(model, images_list, xz, input_seqs, hanning, bs=bs)

