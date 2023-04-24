import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)
env_path = os.path.join(os.path.dirname(__file__), '../../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.VOT2020_super_only_mask_384_HP.seqtrack_alpha_seg_class import run_vot_exp

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
run_vot_exp('seqtrack', 'seqtrack_l384',
            'ARcm_coco_seg_only_mask_384', 0.65, VIS=False)