# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOTS20 sequence dataset.
"""
import csv
import os
import os.path as osp
from argparse import Namespace
from typing import Optional, Tuple

import numpy as np
import pycocotools.mask as rletools

from .mot17_sequence import MOT17Sequence


class MOTS20Sequence(MOT17Sequence):
    """Multiple Object and Segmentation Tracking (MOTS20) Dataset.

    This dataloader is designed so that it can handle only one sequence,
    if more have to be handled one should inherit from this class.
    """
    data_folder = 'MOTS20'

    def __init__(self, root_dir: str = 'data', seq_name: Optional[str] = None,
                 vis_threshold: float = 0.0, img_transform: Namespace = None) -> None:
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons
                                   above which they are selected
        """
        super().__init__(root_dir, seq_name, None, vis_threshold, img_transform)

    def get_track_boxes_and_visbility(self) -> Tuple[dict, dict]:
        boxes = {}
        visibility = {}

        for i in range(1, self.seq_length + 1):
            boxes[i] = {}
            visibility[i] = {}

        gt_file = self.get_gt_file_path()
        if not osp.exists(gt_file):
            return boxes, visibility

        mask_objects_per_frame = load_mots_gt(gt_file)
        for frame_id, mask_objects in mask_objects_per_frame.items():
            for mask_object in mask_objects:
                # class_id = 1 is car
                # class_id = 2 is pedestrian
                # class_id = 10 IGNORE
                if mask_object.class_id in [1, 10]:
                    continue

                bbox = rletools.toBbox(mask_object.mask)
                x1, y1, w, h = [int(c) for c in bbox]
                bbox = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)

                # area = bbox[2] * bbox[3]
                # image_id = img_file_name_to_id[f"{seq}_{frame_id:06d}.jpg"]

                # segmentation = {
                #     'size': mask_object.mask['size'],
                #     'counts': mask_object.mask['counts'].decode(encoding='UTF-8')}

                boxes[frame_id][mask_object.track_id] = bbox
                visibility[frame_id][mask_object.track_id] = 1.0

        return boxes, visibility

    def write_results(self, results: dict, output_dir: str) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_file_path = osp.join(output_dir, f"{self._seq_name}.txt")

        with open(result_file_path, "w") as res_file:
            writer = csv.writer(res_file, delimiter=' ')
            for i, track in results.items():
                for frame, data in track.items():
                    mask = np.asfortranarray(data['mask'])
                    rle_mask = rletools.encode(mask)

                    writer.writerow([
                        frame + 1,
                        i + 1,
                        2,  # class pedestrian
                        mask.shape[0],
                        mask.shape[1],
                        rle_mask['counts'].decode(encoding='UTF-8')])

    def load_results(self, results_dir: str) -> dict:
        results = {}

        if results_dir is None:
            return results

        file_path = osp.join(results_dir, self.results_file_name)

        if not os.path.isfile(file_path):
            return results

        mask_objects_per_frame = load_mots_gt(file_path)

        for frame_id, mask_objects in mask_objects_per_frame.items():
            for mask_object in mask_objects:
                # class_id = 1 is car
                # class_id = 2 is pedestrian
                # class_id = 10 IGNORE
                if mask_object.class_id in [1, 10]:
                    continue

                bbox = rletools.toBbox(mask_object.mask)
                x1, y1, w, h = [int(c) for c in bbox]
                bbox = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)

                # area = bbox[2] * bbox[3]
                # image_id = img_file_name_to_id[f"{seq}_{frame_id:06d}.jpg"]

                # segmentation = {
                #     'size': mask_object.mask['size'],
                #     'counts': mask_object.mask['counts'].decode(encoding='UTF-8')}

                track_id = mask_object.track_id - 1
                if track_id not in results:
                    results[track_id] = {}

                results[track_id][frame_id - 1] = {}
                results[track_id][frame_id - 1]['mask'] = rletools.decode(mask_object.mask)
                results[track_id][frame_id - 1]['bbox'] = bbox.tolist()
                results[track_id][frame_id - 1]['score'] = 1.0

        return results

    def __str__(self) -> str:
        return self._seq_name


class SegmentedObject:
    """
    Helper class for segmentation objects.
    """
    def __init__(self, mask: dict, class_id: int, track_id: int) -> None:
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


def load_mots_gt(path: str) -> dict:
    """Load MOTS ground truth from path."""
    objects_per_frame = {}
    track_ids_per_frame = {}  # Check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # Check that no frame contains overlapping masks

    with open(path, "r") as gt_file:
        for line in gt_file:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, f"Multiple objects with track id {fields[1]} in frame {fields[0]}"
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            if not(class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            mask = {
                'size': [int(fields[3]), int(fields[4])],
                'counts': fields[5].encode(encoding='UTF-8')}
            if frame not in combined_mask_per_frame:
                combined_mask_per_frame[frame] = mask
            elif rletools.area(rletools.merge([
                    combined_mask_per_frame[frame], mask],
                    intersect=True)):
                assert False, "Objects with overlapping masks in frame " + fields[0]
            else:
                combined_mask_per_frame[frame] = rletools.merge(
                    [combined_mask_per_frame[frame], mask],
                    intersect=False)
            objects_per_frame[frame].append(SegmentedObject(
                mask,
                class_id,
                int(fields[1])
            ))

    return objects_per_frame
