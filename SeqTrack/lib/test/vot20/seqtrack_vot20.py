from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import cv2
import torch
# import vot
import sys
import time
import os
from lib.test.evaluation import Tracker
import lib.test.vot20.vot as vot
from lib.test.vot20.vot20_utils import *

'''seqtrack for vot20'''


class SEQTRACK(object):
    def __init__(self, tracker_name='seqtrack', para_name='seqtrack_b256'):
        # create tracker
        tracker_info = Tracker(tracker_name, para_name, "vot20", None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params)


    def initialize(self, img_rgb, selection):
        # init on the 1st frame
        # region = rect_from_mask(mask)
        x, y, w, h = selection
        bbox = [x,y,w,h]
        self.H, self.W, _ = img_rgb.shape
        self.init_info = {'init_bbox': bbox}
        _ = self.tracker.initialize(img_rgb, self.init_info)

    def track(self, img_rgb):
        # track
        outputs = self.tracker.track(img_rgb, self.init_info)
        pred_bbox = outputs['target_bbox']
        return pred_bbox


def run_vot_exp(tracker_name, para_name, vis=False):

    torch.set_num_threads(1)
    save_root = os.path.join('/path_to_SeqTrack/external/vot20/vis', para_name)
    if vis and (not os.path.exists(save_root)):
        os.makedirs(save_root)
    tracker = SEQTRACK(tracker_name=tracker_name, para_name=para_name)
    handle = vot.VOT("rectangle")
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    if vis:
        '''for vis'''
        seq_name = imagefile.split('/')[-3]
        save_v_dir = os.path.join(save_root,seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB) # Right
    # mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
    # mask = make_full_size(selection, (image.shape[1], image.shape[0]))
    tracker.initialize(image, selection)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        b1 = tracker.track(image)
        handle.report(vot.Rectangle(*b1))
        if vis:
            '''Visualization'''
            # original image
            image_ori = image[:,:,::-1].copy() # RGB --> BGR
            image_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, image_name)
            # cv2.imwrite(save_path, image_ori)
            # tracker box
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg','_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)

