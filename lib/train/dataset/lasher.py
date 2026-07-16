import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame


class LasHeR(BaseVideoDataset):
    """ LasHeR dataset(aligned version).

    Publication:
        A Large-scale High-diversity Benchmark for RGBT Tracking
        Chenglong Li, Wanlin Xue, Yaqing Jia, Zhichen Qu, Bin Luo, Jin Tang, and Dengdi Sun
        https://arxiv.org/pdf/2104.13202.pdf

    Download dataset from https://github.com/BUGPLEASEOUT/LasHeR
    """

    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the LasHeR trainingset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasher_dir if root is None else root
        assert split in ['train', 'val','all'], 'Only support all, train or val split in LasHeR, got {}'.format(split)
        super().__init__('LasHeR', root)
        self.dtype = dtype

        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        # Filter out sequences that don't exist or have missing files
        valid_sequences = []
        for seq_name in self.sequence_list:
            seq_dir = os.path.join(root, seq_name)
            init_file = os.path.join(seq_dir, 'init.txt')
            if os.path.isdir(seq_dir) and os.path.isfile(init_file):
                valid_sequences.append(seq_name)
            else:
                print("WARNING: Skipping sequence '%s' (missing directory or init.txt)" % seq_name)
        skipped = len(self.sequence_list) - len(valid_sequences)
        if skipped > 0:
            print("Filtered out %d/%d invalid sequences from LasHeR-%s dataset" % (skipped, len(self.sequence_list), split))
        self.sequence_list = valid_sequences

        # seq_id is the index of the folder inside the got10k root path
        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        return 'lasher'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True # w=h=0 in visible.txt and infrared.txt is occlusion/oov

    def _get_sequence_list(self, split):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        file_path = os.path.join(ltr_path, 'data_specs', 'lasher_{}.txt'.format(split))
        with open(file_path, 'r') as f:
            dir_list = f.read().splitlines()
        return dir_list

    def _read_bb_anno(self, seq_path):
        # in lasher dataset, visible.txt is same as infrared.txt
        rgb_bb_anno_file = os.path.join(seq_path, "init.txt")
        if not os.path.exists(rgb_bb_anno_file):
            raise FileNotFoundError("Missing annotation file: %s" % rgb_bb_anno_file)
        rgb_gt = pandas.read_csv(rgb_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(rgb_gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        """2022/8/10 ir and rgb have synchronous w=h=0 frame_index"""
        seq_path = self._get_sequence_path(seq_id)
        try:
            bbox = self._read_bb_anno(seq_path)
        except (FileNotFoundError, pandas.errors.EmptyDataError) as e:
            print("WARNING: Skipping sequence %s (seq_id=%d): %s" % (self.sequence_list[seq_id], seq_id, e))
            return None
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()

        # # 新增读取文本标注的代码
        # text_label_path = os.path.join(seq_path, 'description.txt')
        # try:
        #     with open(text_label_path, 'r', encoding='utf-8') as f:
        #         text_label = f.read().strip()  # 读取整个文件内容作为文本标注
        # except FileNotFoundError:
        #     text_label = None  # 如果文件不存在，可以设置为默认值或进行其他处理
        #     print('FileNotFound:' + text_label_path)
        # return {'bbox': bbox, 'valid': valid, 'visible': visible, 'text_label': text_label}

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # Note original filename is chaotic, we rename them
        visible_dir = os.path.join(seq_path, "visible")
        infrared_dir = os.path.join(seq_path, "infrared")
        if not os.path.isdir(visible_dir) or not os.path.isdir(infrared_dir):
            raise FileNotFoundError("Missing visible/infrared directory in %s" % seq_path)
        rgb_frame_path = sorted(os.listdir(visible_dir))  # frames start from 0
        ir_frame_path = sorted(os.listdir(infrared_dir))

        return os.path.join(visible_dir, rgb_frame_path[frame_id]), os.path.join(infrared_dir, ir_frame_path[frame_id])  # jpg jpg

    def _get_frame(self, seq_path, frame_id):
        try:
            rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id)
        except (FileNotFoundError, IndexError) as e:
            raise FileNotFoundError("Cannot get frame %d in %s: %s" % (frame_id, seq_path, e))
        img = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        return img  # (h,w,6)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Skip sequences with missing annotation files
        if anno is None:
            return None, None, None

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        anno_frames = {}
        for key, value in anno.items():
            # if key == 'text_label':  # 特别处理文本标注
            #     anno_frames[key] = value  # 文本标注是整个序列的全局标注，直接传递
            # else:
            #     anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
