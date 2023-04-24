import torch.nn as nn
import torch
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from torch.nn import functional as F
from ltr.models.neck.neck_utils import *

class CorrNL(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, pool_size=8, use_NL=True):
        super().__init__()
        self.prroi_pool = PrRoIPool2D(pool_size, pool_size, 1/16)
        num_corr_channel = pool_size*pool_size
        self.channel_attention = SEModule(num_corr_channel,reduction=4)
        self.spatial_attention = NONLocalBlock2D(in_channels=num_corr_channel)
        self.use_NL = use_NL
    def forward(self, feat1, feat2, bb1):
        """Runs the ATOM IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4)."""

        assert bb1.dim() == 3
        # num_images, num_sequences = bb1.size()[:2] # 1, 64

        # Extract first train sample
        if len(feat1)==1:
            feat1 = feat1[0] # size为(64,C,H,W)
            feat2 = feat2[0] # size为(64,C,H,W)
            bb1 = bb1[0,...] # (64,4)
        else:
            raise ValueError("Only support single-layer feature map")
        '''get PrRoIPool feature '''
        # Add batch_index to rois
        batch_size = bb1.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb1.device) # (64,1)
        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb1 = bb1.clone()
        bb1[:, 2:4] = bb1[:, 0:2] + bb1[:, 2:4]
        roi1 = torch.cat((batch_index, bb1), dim=1) #(64,1),(64,4) ---> (64,5)
        feat_roi1 = self.prroi_pool(feat1, roi1) # (64,C,H,W)
        feat_corr,_ = self.corr_fun(feat_roi1, feat2)
        # print('相关后的特征维度是:',feat_corr.size())#(batch,StxSt,Sr,Sr)
        '''channel attention: Squeeze and Excitation'''
        feat_ca = self.channel_attention(feat_corr) # 计算通道注意力特征
        '''spatial attention: Non-local 2D'''
        feat_sa = self.spatial_attention(feat_ca)
        return feat_sa

    def get_ref_kernel(self, feat1, bb1):
        assert bb1.dim() == 3
        # num_images, num_sequences = bb1.size()[:2] # 1, 64

        # Extract first train sample
        if len(feat1) == 1:
            feat1 = feat1[0]  # size为(64,C,H,W)
            bb1 = bb1[0, ...]  # (64,4)
        else:
            raise ValueError("Only support single-layer feature map")
        '''get PrRoIPool feature '''
        # Add batch_index to rois
        batch_size = bb1.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb1.device)  # (64,1)
        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb1 = bb1.clone()
        bb1[:, 2:4] = bb1[:, 0:2] + bb1[:, 2:4]
        roi1 = torch.cat((batch_index, bb1), dim=1)  # (64,1),(64,4) ---> (64,5)
        '''注意: feat1 and roi1 must be cuda tensor'''
        self.ref_kernel = self.prroi_pool(feat1.float(), roi1)  # (64,C,H,W)
        # self.ref_kernel.half()

    def fuse_feat(self, feat2):
        '''fuse features from reference and test branch'''
        if len(feat2) == 1:
            feat2 = feat2[0]
        '''Step1: pixel-wise correlation'''
        feat_corr,_ = self.corr_fun(self.ref_kernel, feat2)
        # print('相关后的特征维度是:',feat_corr.size())#(batch,StxSt,Sr,Sr) (batch,64,16,16)
        '''Step2: channel attention: Squeeze and Excitation'''
        feat_ca = self.channel_attention(feat_corr) # 计算通道注意力特征
        if not self.use_NL:
            # print('not use non-local')
            return feat_ca
        else:
            '''Step3: spatial attention: Non-local 2D'''
            feat_sa = self.spatial_attention(feat_ca)
            return feat_sa


    def corr_fun(self, Kernel_tmp, Feature, KERs=None):
        size = Kernel_tmp.size()
        CORR = []
        Kernel = []
        for i in range(len(Feature)):
            ker = Kernel_tmp[i:i + 1]
            fea = Feature[i:i + 1]
            ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)
            ker = ker.unsqueeze(2).unsqueeze(3)
            if not (type(KERs) == type(None)):
                ker = torch.cat([ker, KERs[i]], 0)
            co = F.conv2d(fea, ker.contiguous())
            CORR.append(co)
            ker = ker.unsqueeze(0)
            Kernel.append(ker)
        corr = torch.cat(CORR, 0)
        Kernel = torch.cat(Kernel, 0)
        return corr, Kernel
