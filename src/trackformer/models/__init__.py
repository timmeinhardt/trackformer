# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from .backbone import build_backbone
from .deformable_detr import DeformableDETR, DeformablePostProcess
from .deformable_transformer import build_deforamble_transformer
from .detr import DETR, PostProcess, SetCriterion
from .detr_segmentation import (DeformableDETRSegm, DeformableDETRSegmTracking,
                                DETRSegm, DETRSegmTracking,
                                PostProcessPanoptic, PostProcessSegm)
from .detr_tracking import DeformableDETRTracking, DETRTracking
from .matcher import build_matcher
from .transformer import build_transformer


def build_model(args):
    if args.dataset == 'coco':
        num_classes = 91
    elif args.dataset == 'coco_panoptic':
        num_classes = 250
    elif args.dataset in ['coco_person', 'mot', 'mot_crowdhuman', 'crowdhuman', 'mot_coco_person']:
        # num_classes = 91
        num_classes = 20
        # num_classes = 1
    else:
        raise NotImplementedError

    device = torch.device(args.device)
    backbone = build_backbone(args)
    matcher = build_matcher(args)

    detr_kwargs = {
        'backbone': backbone,
        'num_classes': num_classes - 1 if args.focal_loss else num_classes,
        'num_queries': args.num_queries,
        'aux_loss': args.aux_loss,
        'overflow_boxes': args.overflow_boxes}

    tracking_kwargs = {
        'track_query_false_positive_prob': args.track_query_false_positive_prob,
        'track_query_false_negative_prob': args.track_query_false_negative_prob,
        'matcher': matcher,
        'backprop_prev_frame': args.track_backprop_prev_frame,}

    mask_kwargs = {
        'freeze_detr': args.freeze_detr}

    if args.deformable:
        transformer = build_deforamble_transformer(args)

        detr_kwargs['transformer'] = transformer
        detr_kwargs['num_feature_levels'] = args.num_feature_levels
        detr_kwargs['with_box_refine'] = args.with_box_refine
        detr_kwargs['two_stage'] = args.two_stage
        detr_kwargs['multi_frame_attention'] = args.multi_frame_attention
        detr_kwargs['multi_frame_encoding'] = args.multi_frame_encoding
        detr_kwargs['merge_frame_features'] = args.merge_frame_features

        if args.tracking:
            if args.masks:
                model = DeformableDETRSegmTracking(mask_kwargs, tracking_kwargs, detr_kwargs)
            else:
                model = DeformableDETRTracking(tracking_kwargs, detr_kwargs)
        else:
            if args.masks:
                model = DeformableDETRSegm(mask_kwargs, detr_kwargs)
            else:
                model = DeformableDETR(**detr_kwargs)
    else:
        transformer = build_transformer(args)

        detr_kwargs['transformer'] = transformer

        if args.tracking:
            if args.masks:
                model = DETRSegmTracking(mask_kwargs, tracking_kwargs, detr_kwargs)
            else:
                model = DETRTracking(tracking_kwargs, detr_kwargs)
        else:
            if args.masks:
                model = DETRSegm(mask_kwargs, detr_kwargs)
            else:
                model = DETR(**detr_kwargs)

    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef,}

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses.append('masks')

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tracking=args.tracking,
        track_query_false_positive_eos_weight=args.track_query_false_positive_eos_weight,)
    criterion.to(device)

    if args.focal_loss:
        postprocessors = {'bbox': DeformablePostProcess()}
    else:
        postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
