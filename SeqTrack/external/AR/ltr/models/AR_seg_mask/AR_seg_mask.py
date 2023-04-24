import torch.nn as nn
from ltr.models.neck import CorrNL
from ltr import model_constructor
import torch
import ltr.models.backbone.resnet_seg as resnet_seg

from ltr.models.head import seg_network
from easydict import EasyDict as edict

'''2020.4.14 replace mask head with frtm for higher-quality mask'''
'''2020.4.22 Only use the mask branch'''


class ARnet_seg_mask(nn.Module):
    """ Scale Estimation network module with three branches: bbox, coner and mask. """
    def __init__(self, feature_extractor, neck_module, head_module, used_layers,
                 extractor_grad=True,output_size=(256,256)):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(ARnet_seg_mask, self).__init__()

        self.feature_extractor = feature_extractor
        self.neck = neck_module
        self.refiner = head_module
        self.used_layers = used_layers
        self.output_size = output_size

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, mode='train'):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        self.forward_ref(train_imgs, train_bb)
        pred_dict = self.forward_test(test_imgs, mode)
        return pred_dict

    def forward_ref(self, train_imgs, train_bb):
        """ Forward pass of reference branch.
        size of train_imgs is (1,batch,3,H,W), train_bb is (1,batch,4)"""
        num_sequences = train_imgs.shape[-4] # batch
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1 # 1

        # Extract backbone features
        '''train_feat OrderedDict, key:'layer4' '''
        train_feat_dict = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:])) # 输入size是(batch,3,256,256)

        train_feat_list = [feat for feat in train_feat_dict.values()] #list,其中每个元素对应一层输出的特征(tensor)

        # get reference feature
        self.neck.get_ref_kernel(train_feat_list, train_bb.view(num_train_images, num_sequences, 4))


    def forward_test(self, test_imgs, mode='train'):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        output = {}
        # Extract backbone features
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                        layers=['layer1','layer2','layer3','layer4','layer5'])# 输入size是(batch,3,256,256)
        '''list,tensor'''
        # Save low-level feature list
        # Lfeat_list = [feat for name, feat in test_feat_dict.items() if name != 'layer3']

        # fuse feature from two branches
        fusion_feat = self.neck.fuse_feat([test_feat_dict['layer4']])
        # Obtain bbox prediction
        if mode=='train':
            output['mask'] = torch.sigmoid(self.refiner(fusion_feat, test_feat_dict, self.output_size))
        elif mode == 'mask':
            output = torch.sigmoid(self.refiner(fusion_feat, test_feat_dict, self.output_size))
        else:
            raise ValueError("mode should be train or test")
        return output

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.used_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)



@model_constructor
def ARnet_seg_mask_resnet50(backbone_pretrained=True,used_layers=('layer4',),pool_size=None):
    # backbone
    backbone_net = resnet_seg.resnet50(pretrained=backbone_pretrained)
    # neck
    neck_net = CorrNL.CorrNL(pool_size=pool_size)
    # multiple heads
    '''create segnet'''
    in_channels = 1024
    # disc_params = edict(layer="layer4", in_channels=in_channels, c_channels=96, out_channels=64) # non-local feat (64 channels rather than 1)
    '''2020.4.22 change "out_channels" to pool_size * pool_size'''
    disc_params = edict(layer="layer4", in_channels=in_channels, c_channels=96, out_channels=pool_size*pool_size) # non-local feat (64 channels rather than 1)
    refnet_params = edict(
        layers=("layer5", "layer4", "layer3", "layer2"),
        nchannels=64, use_batch_norm=True)
    disc_params.in_channels = backbone_net.get_out_channels()[disc_params.layer]

    p = refnet_params
    refinement_layers_channels = {L: nch for L, nch in backbone_net.get_out_channels().items() if L in p.layers}
    refiner = seg_network.SegNetwork(disc_params.out_channels, p.nchannels, refinement_layers_channels, p.use_batch_norm)
    '''create Alpha-Refine'''
    net = ARnet_seg_mask(feature_extractor=backbone_net, neck_module=neck_net,
                         head_module=refiner,
                         used_layers=used_layers, extractor_grad=True,
                         output_size=(int(pool_size*2*16),int(pool_size*2*16)))
    return net
