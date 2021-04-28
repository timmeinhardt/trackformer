# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import json
import numpy as np


LOG_DIR = 'logs/visdom'

METRICS = ['MOTA', 'IDF1', 'BBOX AP IoU=0.50:0.95', 'MASK AP IoU=0.50:0.95']

RUNS = [
    'mot17_train_1_deformable_full_res',
    'mot17_train_2_deformable_full_res',
    'mot17_train_3_deformable_full_res',
    'mot17_train_4_deformable_full_res',
    'mot17_train_5_deformable_full_res',
    'mot17_train_6_deformable_full_res',
    'mot17_train_7_deformable_full_res',
    ]

RUNS = [
    'mot17_train_1_no_pretrain_deformable_tracking',
    'mot17_train_2_no_pretrain_deformable_tracking',
    'mot17_train_3_no_pretrain_deformable_tracking',
    'mot17_train_4_no_pretrain_deformable_tracking',
    'mot17_train_5_no_pretrain_deformable_tracking',
    'mot17_train_6_no_pretrain_deformable_tracking',
    'mot17_train_7_no_pretrain_deformable_tracking',
    ]

RUNS = [
    'mot17_train_1_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_2_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_3_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_4_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_5_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_6_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_7_coco_pretrain_deformable_tracking_lr=0.00001',
    ]

RUNS = [
    'mot17_train_1_crowdhuman_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_2_crowdhuman_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_3_crowdhuman_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_4_crowdhuman_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_5_crowdhuman_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_6_crowdhuman_coco_pretrain_deformable_tracking_lr=0.00001',
    'mot17_train_7_crowdhuman_coco_pretrain_deformable_tracking_lr=0.00001',
    ]

# RUNS = [
#     'mot17_train_1_no_pretrain_deformable_tracking_eos_coef=0.2',
#     'mot17_train_2_no_pretrain_deformable_tracking_eos_coef=0.2',
#     'mot17_train_3_no_pretrain_deformable_tracking_eos_coef=0.2',
#     'mot17_train_4_no_pretrain_deformable_tracking_eos_coef=0.2',
#     'mot17_train_5_no_pretrain_deformable_tracking_eos_coef=0.2',
#     'mot17_train_6_no_pretrain_deformable_tracking_eos_coef=0.2',
#     'mot17_train_7_no_pretrain_deformable_tracking_eos_coef=0.2',
#     ]

# RUNS = [
#     'mot17_train_1_no_pretrain_deformable_tracking_lr_drop=50',
#     'mot17_train_2_no_pretrain_deformable_tracking_lr_drop=50',
#     'mot17_train_3_no_pretrain_deformable_tracking_lr_drop=50',
#     'mot17_train_4_no_pretrain_deformable_tracking_lr_drop=50',
#     'mot17_train_5_no_pretrain_deformable_tracking_lr_drop=50',
#     'mot17_train_6_no_pretrain_deformable_tracking_lr_drop=50',
#     'mot17_train_7_no_pretrain_deformable_tracking_lr_drop=50',
#     ]

# RUNS = [
#     'mot17_train_1_no_pretrain_deformable_tracking_save_model_interval=1',
#     'mot17_train_2_no_pretrain_deformable_tracking_save_model_interval=1',
#     'mot17_train_3_no_pretrain_deformable_tracking_save_model_interval=1',
#     'mot17_train_4_no_pretrain_deformable_tracking_save_model_interval=1',
#     'mot17_train_5_no_pretrain_deformable_tracking_save_model_interval=1',
#     'mot17_train_6_no_pretrain_deformable_tracking_save_model_interval=1',
#     'mot17_train_7_no_pretrain_deformable_tracking_save_model_interval=1',
#     ]

# RUNS = [
    # 'mot17_train_1_no_pretrain_deformable_tracking_save_model_interval=1',
    # 'mot17_train_2_no_pretrain_deformable_tracking_save_model_interval=1',
    # 'mot17_train_3_no_pretrain_deformable_tracking_save_model_interval=1',
    # 'mot17_train_4_no_pretrain_deformable_tracking_save_model_interval=1',
    # 'mot17_train_5_no_pretrain_deformable_tracking_save_model_interval=1',
    # 'mot17_train_6_no_pretrain_deformable_tracking_save_model_interval=1',
    # 'mot17_train_7_no_pretrain_deformable_tracking_save_model_interval=1',
    # ]

# RUNS = [
#     'mot17_train_1_no_pretrain_deformable_full_res',
#     'mot17_train_2_no_pretrain_deformable_full_res',
#     'mot17_train_3_no_pretrain_deformable_full_res',
#     'mot17_train_4_no_pretrain_deformable_full_res',
#     'mot17_train_5_no_pretrain_deformable_full_res',
#     'mot17_train_6_no_pretrain_deformable_full_res',
#     'mot17_train_7_no_pretrain_deformable_full_res',
#     ]

# RUNS = [
#     'mot17_train_1_no_pretrain_deformable_tracking_track_query_false_positive_eos_weight=False',
#     'mot17_train_2_no_pretrain_deformable_tracking_track_query_false_positive_eos_weight=False',
#     'mot17_train_3_no_pretrain_deformable_tracking_track_query_false_positive_eos_weight=False',
#     'mot17_train_4_no_pretrain_deformable_tracking_track_query_false_positive_eos_weight=False',
#     'mot17_train_5_no_pretrain_deformable_tracking_track_query_false_positive_eos_weight=False',
#     'mot17_train_6_no_pretrain_deformable_tracking_track_query_false_positive_eos_weight=False',
#     'mot17_train_7_no_pretrain_deformable_tracking_track_query_false_positive_eos_weight=False',
#     ]

# RUNS = [
#     'mot17_train_1_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0',
#     'mot17_train_2_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0',
#     'mot17_train_3_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0',
#     'mot17_train_4_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0',
#     'mot17_train_5_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0',
#     'mot17_train_6_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0',
#     'mot17_train_7_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0',
#     ]

# RUNS = [
#     'mot17_train_1_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0',
#     'mot17_train_2_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0',
#     'mot17_train_3_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0',
#     'mot17_train_4_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0',
#     'mot17_train_5_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0',
#     'mot17_train_6_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0',
#     'mot17_train_7_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0',
#     ]

# RUNS = [
#     'mot17_train_1_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0_track_query_false_negative_prob=0_0',
#     'mot17_train_2_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0_track_query_false_negative_prob=0_0',
#     'mot17_train_3_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0_track_query_false_negative_prob=0_0',
#     'mot17_train_4_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0_track_query_false_negative_prob=0_0',
#     'mot17_train_5_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0_track_query_false_negative_prob=0_0',
#     'mot17_train_6_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0_track_query_false_negative_prob=0_0',
#     'mot17_train_7_no_pretrain_deformable_tracking_track_query_false_positive_prob=0_0_track_prev_frame_range=0_track_query_false_negative_prob=0_0',
#     ]

# RUNS = [
#     'mot17_train_1_no_pretrain_deformable',
#     'mot17_train_2_no_pretrain_deformable',
#     'mot17_train_3_no_pretrain_deformable',
#     'mot17_train_4_no_pretrain_deformable',
#     'mot17_train_5_no_pretrain_deformable',
#     'mot17_train_6_no_pretrain_deformable',
#     'mot17_train_7_no_pretrain_deformable',
#     ]

#
# MOTS 4-fold split
#

# RUNS = [
#     'mots20_train_1_coco_tracking',
#     'mots20_train_2_coco_tracking',
#     'mots20_train_3_coco_tracking',
#     'mots20_train_4_coco_tracking',
#     ]

# RUNS = [
#     'mots20_train_1_coco_tracking_full_res_masks=False',
#     'mots20_train_2_coco_tracking_full_res_masks=False',
#     'mots20_train_3_coco_tracking_full_res_masks=False',
#     'mots20_train_4_coco_tracking_full_res_masks=False',
#     ]

# RUNS = [
#     'mots20_train_1_coco_full_res_pretrain_masks=False_lr_0_0001',
#     'mots20_train_2_coco_full_res_pretrain_masks=False_lr_0_0001',
#     'mots20_train_3_coco_full_res_pretrain_masks=False_lr_0_0001',
#     'mots20_train_4_coco_full_res_pretrain_masks=False_lr_0_0001',
#     ]

# RUNS = [
#     'mots20_train_1_coco_tracking_full_res_masks=False_pretrain',
#     'mots20_train_2_coco_tracking_full_res_masks=False_pretrain',
#     'mots20_train_3_coco_tracking_full_res_masks=False_pretrain',
#     'mots20_train_4_coco_tracking_full_res_masks=False_pretrain',
#     ]

# RUNS = [
#     'mot17det_train_1_mots_track_bbox_proposals_pretrain_train_1_mots_vis_save_model_interval_1',
#     'mot17det_train_2_mots_track_bbox_proposals_pretrain_train_3_mots_vis_save_model_interval_1',
#     'mot17det_train_3_mots_track_bbox_proposals_pretrain_train_4_mots_vis_save_model_interval_1',
#     'mot17det_train_4_mots_track_bbox_proposals_pretrain_train_6_mots_vis_save_model_interval_1',
# ]

if __name__ == '__main__':
    results = {}

    for r in RUNS:
        print(r)
        log_file = os.path.join(LOG_DIR, f"{r}.json")

        with open(log_file) as json_file:
            data = json.load(json_file)

            window = [
                window for window in data['jsons'].values()
                if window['title'] == 'VAL EVAL EPOCHS'][0]

            for m in METRICS:
                if m not in window['legend']:
                    continue
                elif m not in results:
                    results[m] = []

                idxs = window['legend'].index(m)

                values = window['content']['data'][idxs]['y']
                results[m].append(values)

        print(f'NUM EPOCHS: {len(values)}')

    min_length = min([len(l) for l in next(iter(results.values()))])

    for metric in results.keys():
        results[metric] = [l[:min_length] for l in results[metric]]

    mean_results = {
        metric: np.array(results[metric]).mean(axis=0)
        for metric in results.keys()}

    print("* METRIC INTERVAL = BEST EPOCHS")
    for metric in results.keys():
        best_interval = mean_results[metric].argmax()
        print(mean_results[metric])
        print(
            f'{metric}: {mean_results[metric].max():.2%} at {best_interval + 1}/{len(mean_results[metric])} '
            f'{[(mmetric, f"{mean_results[mmetric][best_interval]:.2%}") for mmetric in results.keys() if not mmetric == metric]}')
