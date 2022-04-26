# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from itertools import product

import numpy as np

from track import ex


if __name__ == "__main__":


    # configs = [
    #     {'dataset_name': ["MOT17-02-FRCNN", "MOT17-10-FRCNN", "MOT17-13-FRCNN"],
    #      'obj_detect_checkpoint_file': 'models/mot17det_train_cross_val_1_mots_vis_track_bbox_proposals_track_encoding_bbox_proposals_prev_frame_5/checkpoint_best_MOTA.pth'},
    #     {'dataset_name': ["MOT17-04-FRCNN", "MOT17-11-FRCNN"],
    #      'obj_detect_checkpoint_file': 'models/mot17det_train_cross_val_2_mots_vis_track_bbox_proposals_track_encoding_bbox_proposals_prev_frame_5/checkpoint_best_MOTA.pth'},
    #     {'dataset_name': ["MOT17-05-FRCNN", "MOT17-09-FRCNN"],
    #      'obj_detect_checkpoint_file': 'models/mot17det_train_cross_val_3_mots_vis_track_bbox_proposals_track_encoding_bbox_proposals_prev_frame_5/checkpoint_best_MOTA.pth'},
    # ]

    # configs = [
    #     {'dataset_name': ["MOT17-02-FRCNN"],
    #      'obj_detect_checkpoint_file': '/storage/user/meinhard/fair_track/models/mot17_train_1_no_pretrain_deformable/checkpoint_best_BBOX_AP_IoU_0_50-0_95.pth'},
    #     {'dataset_name': ["MOT17-04-FRCNN"],
    #      'obj_detect_checkpoint_file': '/storage/user/meinhard/fair_track/models/mot17_train_2_no_pretrain_deformable/checkpoint_best_BBOX_AP_IoU_0_50-0_95.pth'},
    #     {'dataset_name': ["MOT17-05-FRCNN"],
    #      'obj_detect_checkpoint_file': '/storage/user/meinhard/fair_track/models/mot17_train_3_no_pretrain_deformable/checkpoint_best_BBOX_AP_IoU_0_50-0_95.pth'},
    #     {'dataset_name': ["MOT17-09-FRCNN"],
    #      'obj_detect_checkpoint_file': '/storage/user/meinhard/fair_track/models/mot17_train_4_no_pretrain_deformable/checkpoint_best_BBOX_AP_IoU_0_50-0_95.pth'},
    #     {'dataset_name': ["MOT17-10-FRCNN"],
    #      'obj_detect_checkpoint_file': '/storage/user/meinhard/fair_track/models/mot17_train_5_no_pretrain_deformable/checkpoint_best_BBOX_AP_IoU_0_50-0_95.pth'},
    #     {'dataset_name': ["MOT17-11-FRCNN"],
    #      'obj_detect_checkpoint_file': '/storage/user/meinhard/fair_track/models/mot17_train_6_no_pretrain_deformable/checkpoint_best_BBOX_AP_IoU_0_50-0_95.pth'},
    #     {'dataset_name': ["MOT17-13-FRCNN"],
    #      'obj_detect_checkpoint_file': '/storage/user/meinhard/fair_track/models/mot17_train_7_no_pretrain_deformable/checkpoint_best_BBOX_AP_IoU_0_50-0_95.pth'},
    # ]

    # dataset_name = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]

    # general_tracker_cfg = {'public_detections': False, 'reid_sim_only': True, 'reid_greedy_matching': False}
    general_tracker_cfg = {'public_detections': 'min_iou_0_5'}
    # general_tracker_cfg = {'public_detections': False}

    # dataset_name = 'MOT17-TRAIN-FRCNN'
    dataset_name = 'MOT17-TRAIN-ALL'
    # dataset_name = 'MOT20-TRAIN'

    configs = [
        {'dataset_name': dataset_name,

         'frame_range': {'start': 0.5},
         'obj_detect_checkpoint_file': '/storage/user/meinhard/fair_track/models/mot_mot17_train_cross_val_frame_0_0_to_0_5_coco_pretrained_num_queries_500_batch_size=2_num_gpus_7_num_classes_20_AP_det_overflow_boxes_True_prev_frame_rnd_augs_0_2_uniform_false_negative_prob_multi_frame_hidden_dim_288_sep_encoders_batch_queries/checkpoint_epoch_50.pth'},
    ]

    tracker_param_grids = {
        # 'detection_obj_score_thresh': [0.3, 0.4, 0.5, 0.6],
        # 'track_obj_score_thresh': [0.3, 0.4, 0.5, 0.6],
        'detection_obj_score_thresh': [0.4],
        'track_obj_score_thresh': [0.4],
        # 'detection_nms_thresh': [0.95, 0.9, 0.0],
        # 'track_nms_thresh': [0.95, 0.9, 0.0],
        # 'detection_nms_thresh': [0.9],
        # 'track_nms_thresh': [0.9],
        # 'reid_sim_threshold': [0.0, 0.5, 1.0, 10, 50, 100, 200],
        'reid_score_thresh': [0.4],
        # 'inactive_patience': [-1, 5, 10, 20, 30, 40, 50]
        # 'reid_score_thresh': [0.8],
        # 'inactive_patience': [-1],
        # 'inactive_patience': [-1, 5, 10]
        }

    # compute all config combinations
    tracker_param_cfgs = [dict(zip(tracker_param_grids, v))
                          for v in product(*tracker_param_grids.values())]

    # add empty metric arrays
    metrics = ['mota', 'idf1']
    tracker_param_cfgs = [
        {'config': {**general_tracker_cfg, **tracker_cfg}}
        for tracker_cfg in tracker_param_cfgs]

    for m in metrics:
        for tracker_cfg in tracker_param_cfgs:
            tracker_cfg[m] = []

    total_num_experiments = len(tracker_param_cfgs) * len(configs)
    print(f'NUM experiments: {total_num_experiments}')

    # run all tracker config combinations for all experiment configurations
    exp_counter = 1
    for config in configs:
        for tracker_cfg in tracker_param_cfgs:
            print(f"EXPERIMENT: {exp_counter}/{total_num_experiments}")

            config['tracker_cfg'] = tracker_cfg['config']
            run = ex.run(config_updates=config)
            eval_summary = run.result

            for m in metrics:
                tracker_cfg[m].append(eval_summary[m]['OVERALL'])

            exp_counter += 1

    # compute mean for all metrices
    for m in metrics:
        for tracker_cfg in tracker_param_cfgs:
            tracker_cfg[m] = np.array(tracker_cfg[m]).mean()

    for cfg in tracker_param_cfgs:
        print([cfg[m] for m in metrics], cfg['config'])

    # compute and plot best metric config
    for m in metrics:
        best_metric_cfg_idx = np.array(
            [cfg[m] for cfg in tracker_param_cfgs]).argmax()

        print(f"BEST {m.upper()} CFG: {tracker_param_cfgs[best_metric_cfg_idx]['config']}")

    # TODO
    best_mota_plus_idf1_cfg_idx = np.array(
        [cfg['mota'] + cfg['idf1'] for cfg in tracker_param_cfgs]).argmax()
    print(f"BEST MOTA PLUS IDF1 CFG: {tracker_param_cfgs[best_mota_plus_idf1_cfg_idx]['config']}")
