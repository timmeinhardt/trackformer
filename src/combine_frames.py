# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Combine two sets of frames to one.
"""
import os
import os.path as osp

from PIL import Image

OUTPUT_DIR = 'models/mot17_masks_track_rcnn_and_v3_combined'

FRAME_DIR_1 = 'models/mot17_masks_track_rcnn/MOTS20-TEST'
FRAME_DIR_2 = 'models/mot17_masks_v3/MOTS20-ALL'


if __name__ == '__main__':
    seqs_1 = os.listdir(FRAME_DIR_1)
    seqs_2 = os.listdir(FRAME_DIR_2)

    if not osp.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for seq in seqs_1:
        if seq in seqs_2:
            print(seq)
            seg_output_dir = osp.join(OUTPUT_DIR, seq)
            if not osp.exists(seg_output_dir):
                os.makedirs(seg_output_dir)

            frames = os.listdir(osp.join(FRAME_DIR_1, seq))

            for frame in frames:
                img_1 = Image.open(osp.join(FRAME_DIR_1, seq, frame))
                img_2 = Image.open(osp.join(FRAME_DIR_2, seq, frame))

                width = img_1.size[0]
                height = img_2.size[1]

                combined_frame = Image.new('RGB', (width, height * 2))
                combined_frame.paste(img_1, (0, 0))
                combined_frame.paste(img_2, (0, height))

                combined_frame.save(osp.join(seg_output_dir, f'{frame}'))
