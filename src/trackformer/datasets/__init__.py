# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Submodule interface.
"""
from argparse import Namespace
from pycocotools.coco import COCO
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CocoDetection

from .coco import build as build_coco
from .crowdhuman import build_crowdhuman
from .mot import build_mot, build_mot_crowdhuman, build_mot_coco_person


def get_coco_api_from_dataset(dataset: Subset) -> COCO:
    """Return COCO class from PyTorch dataset for evaluation with COCO eval."""
    for _ in range(10):
        # if isinstance(dataset, CocoDetection):
        #     break
        if isinstance(dataset, Subset):
            dataset = dataset.dataset

    if not isinstance(dataset, CocoDetection):
        raise NotImplementedError

    return dataset.coco


def build_dataset(split: str, args: Namespace) -> Dataset:
    """Helper function to build dataset for different splits ('train' or 'val')."""
    if args.dataset == 'coco':
        dataset = build_coco(split, args)
    elif args.dataset == 'coco_person':
        dataset = build_coco(split, args, 'person_keypoints')
    elif args.dataset == 'mot':
        dataset = build_mot(split, args)
    elif args.dataset == 'crowdhuman':
        dataset = build_crowdhuman(split, args)
    elif args.dataset == 'mot_crowdhuman':
        dataset = build_mot_crowdhuman(split, args)
    elif args.dataset == 'mot_coco_person':
        dataset = build_mot_coco_person(split, args)
    elif args.dataset == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        dataset = build_coco_panoptic(split, args)
    else:
        raise ValueError(f'dataset {args.dataset} not supported')

    return dataset
