# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
import time
from os import path as osp

import motmetrics as mm
import numpy as np
import sacred
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from trackformer.datasets.tracking import TrackDatasetFactory
from trackformer.models import build_model
from trackformer.models.tracker import Tracker
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.util.track_utils import (evaluate_mot_accums, get_mot_accum,
                                          interpolate_tracks, plot_sequence)

mm.lap.default_solver = 'lap'

ex = sacred.Experiment('track')
ex.add_config('cfgs/track.yaml')
ex.add_named_config('reid', 'cfgs/track_reid.yaml')


@ex.automain
def main(seed, dataset_name, obj_detect_checkpoint_file, tracker_cfg,
         write_images, output_dir, interpolate, verbose, load_results_dir,
         data_root_dir, generate_attention_maps, frame_range,
         _config, _log, _run, obj_detector_model=None):
    if write_images:
        assert output_dir is not None

    # obj_detector_model is only provided when run as evaluation during
    # training. in that case we omit verbose outputs.
    if obj_detector_model is None:
        sacred.commands.print_config(_run)

    # set all seeds
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    if output_dir is not None:
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        yaml.dump(
            _config,
            open(osp.join(output_dir, 'track.yaml'), 'w'),
            default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    if obj_detector_model is None:
        obj_detect_config_path = os.path.join(
            os.path.dirname(obj_detect_checkpoint_file),
            'config.yaml')
        obj_detect_args = nested_dict_to_namespace(yaml.unsafe_load(open(obj_detect_config_path)))
        img_transform = obj_detect_args.img_transform
        obj_detector, _, obj_detector_post = build_model(obj_detect_args)

        obj_detect_checkpoint = torch.load(
            obj_detect_checkpoint_file, map_location=lambda storage, loc: storage)

        obj_detect_state_dict = obj_detect_checkpoint['model']
        # obj_detect_state_dict = {
        #     k: obj_detect_state_dict[k] if k in obj_detect_state_dict
        #     else v
        #     for k, v in obj_detector.state_dict().items()}

        obj_detect_state_dict = {
            k.replace('detr.', ''): v
            for k, v in obj_detect_state_dict.items()
            if 'track_encoding' not in k}

        obj_detector.load_state_dict(obj_detect_state_dict)
        if 'epoch' in obj_detect_checkpoint:
            _log.info(f"INIT object detector [EPOCH: {obj_detect_checkpoint['epoch']}]")

        obj_detector.cuda()
    else:
        obj_detector = obj_detector_model['model']
        obj_detector_post = obj_detector_model['post']
        img_transform = obj_detector_model['img_transform']

    if hasattr(obj_detector, 'tracking'):
        obj_detector.tracking()

    track_logger = None
    if verbose:
        track_logger = _log.info
    tracker = Tracker(
        obj_detector, obj_detector_post, tracker_cfg,
        generate_attention_maps, track_logger, verbose)

    time_total = 0
    num_frames = 0
    mot_accums = []
    dataset = TrackDatasetFactory(
        dataset_name, root_dir=data_root_dir, img_transform=img_transform)

    for seq in dataset:
        tracker.reset()

        _log.info(f"------------------")
        _log.info(f"TRACK SEQ: {seq}")

        start_frame = int(frame_range['start'] * len(seq))
        end_frame = int(frame_range['end'] * len(seq))

        seq_loader = DataLoader(
            torch.utils.data.Subset(seq, range(start_frame, end_frame)))

        num_frames += len(seq_loader)

        results = seq.load_results(load_results_dir)

        if not results:
            start = time.time()

            for frame_id, frame_data in enumerate(tqdm.tqdm(seq_loader, file=sys.stdout)):
                with torch.no_grad():
                    tracker.step(frame_data)

            results = tracker.get_results()

            time_total += time.time() - start

            _log.info(f"NUM TRACKS: {len(results)} ReIDs: {tracker.num_reids}")
            _log.info(f"RUNTIME: {time.time() - start :.2f} s")

            if interpolate:
                results = interpolate_tracks(results)

            if output_dir is not None:
                _log.info(f"WRITE RESULTS")
                seq.write_results(results, output_dir)
        else:
            _log.info("LOAD RESULTS")

        if seq.no_gt:
            _log.info("NO GT AVAILBLE")
        else:
            mot_accum = get_mot_accum(results, seq_loader)
            mot_accums.append(mot_accum)

            if verbose:
                mot_events = mot_accum.mot_events
                reid_events = mot_events[mot_events['Type'] == 'SWITCH']
                match_events = mot_events[mot_events['Type'] == 'MATCH']

                switch_gaps = []
                for index, event in reid_events.iterrows():
                    frame_id, _ = index
                    match_events_oid = match_events[match_events['OId'] == event['OId']]
                    match_events_oid_earlier = match_events_oid[
                        match_events_oid.index.get_level_values('FrameId') < frame_id]

                    if not match_events_oid_earlier.empty:
                        match_events_oid_earlier_frame_ids = \
                            match_events_oid_earlier.index.get_level_values('FrameId')
                        last_occurrence = match_events_oid_earlier_frame_ids.max()
                        switch_gap = frame_id - last_occurrence
                        switch_gaps.append(switch_gap)

                switch_gaps_hist = None
                if switch_gaps:
                    switch_gaps_hist, _ = np.histogram(
                        switch_gaps, bins=list(range(0, max(switch_gaps) + 10, 10)))
                    switch_gaps_hist = switch_gaps_hist.tolist()

                _log.info(f'SWITCH_GAPS_HIST (bin_width=10): {switch_gaps_hist}')

        if output_dir is not None and write_images:
            _log.info("PLOT SEQ")
            plot_sequence(
                results, seq_loader, osp.join(output_dir, dataset_name, str(seq)),
                write_images, generate_attention_maps)

    if time_total:
        _log.info(f"RUNTIME ALL SEQS (w/o EVAL or IMG WRITE): "
                  f"{time_total:.2f} s for {num_frames} frames "
                  f"({num_frames / time_total:.2f} Hz)")

    if obj_detector_model is None:
        _log.info(f"EVAL:")

        summary, str_summary = evaluate_mot_accums(
            mot_accums,
            [str(s) for s in dataset if not s.no_gt])

        _log.info(f'\n{str_summary}')

        return summary

    return mot_accums
