# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT dataset with tracking training augmentations.
"""
import bisect
import copy
import csv
import os
import random
from pathlib import Path

import torch

from . import transforms as T
from .coco import CocoDetection, make_coco_transforms
from .crowdhuman import build_crowdhuman


class MOT(CocoDetection):

    def __init__(self, img_folder, ann_file, transforms, return_masks,
                 prev_frame=False, prev_frame_range=None, prev_frame_rnd_augs=0.0, norm_transform=None):
        super(MOT, self).__init__(
            img_folder, ann_file, transforms, return_masks, False,
            norm_transform, prev_frame, prev_frame_rnd_augs)

        self._prev_frame_range = prev_frame_range

    @property
    def sequences(self):
        return self.coco.dataset['sequences']

    @property
    def frame_range(self):
        if 'frame_range' in self.coco.dataset:
            return self.coco.dataset['frame_range']
        else:
            return {'start': 0, 'end': 1.0}

    def _add_frame_to_target(self, target, image_id, random_state, key_prefix):
        random.setstate(random_state)
        frame_img, frame_target = self._getitem_from_id(image_id)

        # random jitter
        if self._prev_frame_rnd_augs and random.uniform(0, 1) < 0.5:
            # prev img
            orig_w, orig_h = frame_img.size

            width, height = frame_img.size
            size = random.randint(
                int((1.0 - self._prev_frame_rnd_augs) * min(width, height)),
                int((1.0 + self._prev_frame_rnd_augs) * min(width, height)))
            frame_img, frame_target = T.RandomResize([size])(frame_img, frame_target)

            width, height = frame_img.size
            min_size = (
                int((1.0 - self._prev_frame_rnd_augs) * width),
                int((1.0 - self._prev_frame_rnd_augs) * height))
            transform = T.RandomSizeCrop(min_size=min_size)
            frame_img, frame_target = transform(frame_img, frame_target)

            width, height = frame_img.size
            if orig_w < width:
                frame_img, frame_target = T.RandomCrop((height, orig_w))(frame_img, frame_target)
            else:
                frame_img, frame_target = T.RandomPad(
                    max_size=(orig_w, height))(frame_img, frame_target)

            width, height = frame_img.size
            if orig_h < height:
                frame_img, frame_target = T.RandomCrop((orig_h, width))(frame_img, frame_target)
            else:
                frame_img, frame_target = T.RandomPad(
                    max_size=(width, orig_h))(frame_img, frame_target)

        frame_img, frame_target = self._norm_transforms(frame_img, frame_target)

        target[f'{key_prefix}_image'] = frame_img
        for k, v in frame_target.items():
            target[f'{key_prefix}_{k}'] = v

    def seq_length(self, idx):
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)

    def __getitem__(self, idx):
        random_state = random.getstate()

        img, target = self._getitem_from_id(idx)
        img, target = self._norm_transforms(img, target)

        if self._prev_frame:
            frame_id = self.coco.imgs[idx]['frame_id']

            # first frame has no previous frame
            prev_frame_id = random.randint(
                max(0, frame_id - self._prev_frame_range),
                min(frame_id + self._prev_frame_range, self.seq_length(idx) - 1))
            prev_image_id = self.coco.imgs[idx]['first_frame_image_id'] + prev_frame_id
            self._add_frame_to_target(target, prev_image_id, random_state, 'prev')

        return img, target

    def write_result_files(self, results, output_dir):
        """Write the detections in the format for the MOT17Det sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        """

        files = {}
        for image_id, res in results.items():
            img = self.coco.loadImgs(image_id)[0]
            file_name_without_ext = os.path.splitext(img['file_name'])[0]
            seq_name, frame = file_name_without_ext.split('_')
            frame = int(frame)

            outfile = os.path.join(output_dir, f"{seq_name}.txt")

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                if score <= 0.7:
                    continue
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)


class WeightedConcatDataset(torch.utils.data.ConcatDataset):

    def sample_weight(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        if hasattr(self.datasets[dataset_idx], 'sample_weight'):
            return self.datasets[dataset_idx].sample_weight(sample_idx)
        else:
            return 1 / len(self.datasets[dataset_idx])


def build_mot(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT17Det path {root} does not exist'

    split = getattr(args, f"{image_set}_split")

    img_folder = root / split
    ann_file = root / f"annotations/{split}.json"

    transforms, norm_transforms = make_coco_transforms(
        image_set, args.img_transform)

    dataset = MOT(
        img_folder, ann_file,
        transforms=transforms,
        norm_transform=norm_transforms,
        return_masks=args.masks,
        prev_frame=args.tracking,
        prev_frame_range=args.track_prev_frame_range,
        prev_frame_rnd_augs=args.track_prev_frame_rnd_augs)

    return dataset


def build_mot_crowdhuman(image_set, args):
    if image_set == 'train':
        args_crowdhuman = copy.deepcopy(args)
        args_crowdhuman.train_split = args.crowdhuman_train_split

        crowdhuman_dataset = build_crowdhuman('train', args_crowdhuman)

        if getattr(args, f"{image_set}_split") is None:
            return crowdhuman_dataset

    dataset = build_mot(image_set, args)

    if image_set == 'train':
        dataset = torch.utils.data.ConcatDataset(
            [dataset, crowdhuman_dataset])

    return dataset
