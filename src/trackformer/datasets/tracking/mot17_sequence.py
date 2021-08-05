# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT17 sequence dataset.
"""
import configparser
import csv
import os
import os.path as osp
from argparse import Namespace
from typing import Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..coco import make_coco_transforms
from ..transforms import Compose


class MOT17Sequence(Dataset):
    """Multiple Object Tracking (MOT17) Dataset.

    This dataloader is designed so that it can handle only one sequence,
    if more have to be handled one should inherit from this class.
    """
    data_folder = 'MOT17'

    def __init__(self, root_dir: str = 'data', seq_name: Optional[str] = None,
                 dets: str = '', vis_threshold: float = 0.0, img_transform: Namespace = None) -> None:
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons
                                   above which they are selected
        """
        super().__init__()

        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold

        self._data_dir = osp.join(root_dir, self.data_folder)

        self._train_folders = os.listdir(os.path.join(self._data_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._data_dir, 'test'))

        self.transforms = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))

        self.data = []
        self.no_gt = True
        if seq_name is not None:
            full_seq_name = seq_name
            if self._dets is not None:
                full_seq_name = f"{seq_name}-{dets}"
            assert full_seq_name in self._train_folders or full_seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(full_seq_name)

            self.data = self._sequence()
            self.no_gt = not osp.exists(self.get_gt_file_path())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        width_orig, height_orig = img.size

        img, _ = self.transforms(img)
        width, height = img.size(2), img.size(1)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.tensor([det[:4] for det in data['dets']])
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']
        sample['orig_size'] = torch.as_tensor([int(height_orig), int(width_orig)])
        sample['size'] = torch.as_tensor([int(height), int(width)])

        return sample

    def _sequence(self) -> List[dict]:
        # public detections
        dets = {i: [] for i in range(1, self.seq_length + 1)}
        det_file = self.get_det_file_path()

        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    x1 = float(row[2]) - 1
                    y1 = float(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + float(row[4]) - 1
                    y2 = y1 + float(row[5]) - 1
                    score = float(row[6])
                    bbox = np.array([x1, y1, x2, y2, score], dtype=np.float32)
                    dets[int(float(row[0]))].append(bbox)

        # accumulate total
        img_dir = osp.join(
            self.get_seq_path(),
            self.config['Sequence']['imDir'])

        boxes, visibility = self.get_track_boxes_and_visbility()

        total = [
            {'gt': boxes[i],
             'im_path': osp.join(img_dir, f"{i:06d}.jpg"),
             'vis': visibility[i],
             'dets': dets[i]}
            for i in range(1, self.seq_length + 1)]

        return total

    def get_track_boxes_and_visbility(self) -> Tuple[dict, dict]:
        """ Load ground truth boxes and their visibility."""
        boxes = {}
        visibility = {}

        for i in range(1, self.seq_length + 1):
            boxes[i] = {}
            visibility[i] = {}

        gt_file = self.get_gt_file_path()
        if not osp.exists(gt_file):
            return boxes, visibility

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                # class person, certainity 1
                if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
                    # Make pixel indexes 0-based, should already be 0-based (or not)
                    x1 = int(row[2]) - 1
                    y1 = int(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + int(row[4]) - 1
                    y2 = y1 + int(row[5]) - 1
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

                    frame_id = int(row[0])
                    track_id = int(row[1])

                    boxes[frame_id][track_id] = bbox
                    visibility[frame_id][track_id] = float(row[8])

        return boxes, visibility

    def get_seq_path(self) -> str:
        """ Return directory path of sequence. """
        full_seq_name = self._seq_name
        if self._dets is not None:
            full_seq_name = f"{self._seq_name}-{self._dets}"

        if full_seq_name in self._train_folders:
            return osp.join(self._data_dir, 'train', full_seq_name)
        else:
            return osp.join(self._data_dir, 'test', full_seq_name)

    def get_config_file_path(self) -> str:
        """ Return config file of sequence. """
        return osp.join(self.get_seq_path(), 'seqinfo.ini')

    def get_gt_file_path(self) -> str:
        """ Return ground truth file of sequence. """
        return osp.join(self.get_seq_path(), 'gt', 'gt.txt')

    def get_det_file_path(self) -> str:
        """ Return public detections file of sequence. """
        if self._dets is None:
            return ""

        return osp.join(self.get_seq_path(), 'det', 'det.txt')

    @property
    def config(self) -> dict:
        """ Return config of sequence. """
        config_file = self.get_config_file_path()

        assert osp.exists(config_file), \
            f'Config file does not exist: {config_file}'

        config = configparser.ConfigParser()
        config.read(config_file)
        return config

    @property
    def seq_length(self) -> int:
        """ Return sequence length, i.e, number of frames. """
        return int(self.config['Sequence']['seqLength'])

    def __str__(self) -> str:
        return f"{self._seq_name}-{self._dets}"

    @property
    def results_file_name(self) -> str:
        """ Generate file name of results file. """
        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        if self._dets is None:
            return f"{self._seq_name}.txt"

        return f"{self}.txt"

    def write_results(self, results: dict, output_dir: str) -> None:
        """Write the tracks in the format for MOT16/MOT17 sumbission

        results: dictionary with 1 dictionary for every track with
                 {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_file_path = osp.join(output_dir, self.results_file_name)

        with open(result_file_path, "w") as r_file:
            writer = csv.writer(r_file, delimiter=',')

            for i, track in results.items():
                for frame, data in track.items():
                    x1 = data['bbox'][0]
                    y1 = data['bbox'][1]
                    x2 = data['bbox'][2]
                    y2 = data['bbox'][3]

                    writer.writerow([
                        frame + 1,
                        i + 1,
                        x1 + 1,
                        y1 + 1,
                        x2 - x1 + 1,
                        y2 - y1 + 1,
                        -1, -1, -1, -1])

    def load_results(self, results_dir: str) -> dict:
        results = {}
        if results_dir is None:
            return results

        file_path = osp.join(results_dir, self.results_file_name)

        if not os.path.isfile(file_path):
            return results

        with open(file_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')

            for row in csv_reader:
                frame_id, track_id = int(row[0]) - 1, int(row[1]) - 1

                if track_id not in results:
                    results[track_id] = {}

                x1 = float(row[2]) - 1
                y1 = float(row[3]) - 1
                x2 = float(row[4]) - 1 + x1
                y2 = float(row[5]) - 1 + y1

                results[track_id][frame_id] = {}
                results[track_id][frame_id]['bbox'] = [x1, y1, x2, y2]
                results[track_id][frame_id]['score'] = 1.0

        return results

