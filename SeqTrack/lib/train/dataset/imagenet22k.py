import os
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import torch
import random
from lib.train.dataset.imagenet22k_dataset import IN22KDataset
from collections import OrderedDict
from lib.train.admin import env_settings
import numpy as np


class Imagenet22k(BaseVideoDataset):
    """ The ImageNet22k dataset. ImageNet22k is an image dataset. Thus, we treat each image as a sequence of length 1.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split="train"):
        """
        args:
            root - path to the coco dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            data_fraction (None) - Fraction of images to be used. The images are selected randomly. If None, all the
                                  images  will be used
            split - 'train' or 'val'.
            version - version of coco dataset (2014 or 2017)
        """
        root = env_settings().imagenet22k_dir if root is None else root
        super().__init__('imagenet22k', root, image_loader)

        self.dataset = IN22KDataset(data_root=root, transform=None, fname_format='imagenet5k/{}.JPEG', debug=False)

        # a=1
    # def _get_sequence_list(self):
    #     ann_list = list(self.coco_set.anns.keys())
    #     seq_list = [a for a in ann_list if self.coco_set.anns[a]['iscrowd'] == 0]
    #
    #     return seq_list

    def is_video_sequence(self):
        return False

    # def get_num_classes(self):
    #     return len(self.class_list)

    def get_name(self):
        return 'imagenet22k'

    # def has_class_info(self):
    #     return True

    # def get_class_list(self):
    #     class_list = []
    #     for cat_id in self.cats.keys():
    #         class_list.append(self.cats[cat_id]['name'])
    #     return class_list

    # def has_segmentation_info(self):
    #     return True

    def get_num_sequences(self):
        return len(self.dataset)

    # def _build_seq_per_class(self):
    #     seq_per_class = {}
    #     for i, seq in enumerate(self.sequence_list):
    #         class_name = self.cats[self.coco_set.anns[seq]['category_id']]['name']
    #         if class_name not in seq_per_class:
    #             seq_per_class[class_name] = [i]
    #         else:
    #             seq_per_class[class_name].append(i)
    #
    #     return seq_per_class

    # def get_sequences_in_class(self, class_name):
    #     return self.seq_per_class[class_name]
    #
    def get_sequence_info(self, seq_id):
        '''2021.1.3 To avoid too small bounding boxes. Here we change the threshold to 50 pixels'''
        valid = torch.tensor([True])
        visible = valid.clone().byte()
        return {'bbox': None, 'mask': None, 'valid': valid, 'visible': visible}
    #
    # def _get_anno(self, seq_id):
    #     anno = self.coco_set.anns[self.sequence_list[seq_id]]
    #
    #     return anno

    def _get_frames(self, seq_id):
        img, target = self.dataset.__getitem__(seq_id)
        return img

    # def get_meta_info(self, seq_id):
    #     try:
    #         cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[seq_id]]['category_id']]
    #         object_meta = OrderedDict({'object_class_name': cat_dict_current['name'],
    #                                    'motion_class': None,
    #                                    'major_class': cat_dict_current['supercategory'],
    #                                    'root_class': None,
    #                                    'motion_adverb': None})
    #     except:
    #         object_meta = OrderedDict({'object_class_name': None,
    #                                    'motion_class': None,
    #                                    'major_class': None,
    #                                    'root_class': None,
    #                                    'motion_adverb': None})
    #     return object_meta


    # def get_class_name(self, seq_id):
    #     cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[seq_id]]['category_id']]
    #     return cat_dict_current['name']

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # Imagenet is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        # if anno is None:
        #     anno = self.get_sequence_info(seq_id)

        # anno_frames = {}
        # for key, value in anno.items():
        #     anno_frames[key] = [value[0, ...] for _ in frame_ids]

        # object_meta = self.get_meta_info(seq_id)

        return frame_list, None, None


if __name__ == '__main__':
    data_root = './data/imagenet22k'
    dataset = Imagenet22k(data_root)
