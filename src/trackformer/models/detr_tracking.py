import random

import torch
import torch.nn as nn

from ..util import box_ops
from ..util.misc import NestedTensor
from .deformable_detr import DeformableDETR
from .detr import DETR


class DETRTrackingBase(nn.Module):

    def __init__(self,
                 track_query_false_positive_prob=0.0,
                 track_query_false_negative_prob=0.0,
                 track_query_noise=0.0,
                 matcher=None):
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._track_query_noise = track_query_noise

        self._tracking = False

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True

    def forward(self, samples: NestedTensor, targets: list = None):
        if targets is not None and not self._tracking:
            prev_out, *_ = super().forward([targets[0]['prev_image']])

            prev_outputs_without_aux = {
                k: v for k, v in prev_out.items() if 'aux_outputs' not in k}
            prev_targets = [
                {k.replace('prev_', ''): v for k, v in target.items() if "prev" in k}
                for target in targets]
            prev_indices = self._matcher(prev_outputs_without_aux, prev_targets)

            for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
                prev_out_ind, prev_target_ind = prev_ind

                # random subset
                if self._track_query_false_negative_prob:
                    random_subset_mask = torch.empty(len(prev_target_ind)).uniform_()
                    random_subset_mask = random_subset_mask.ge(
                        self._track_query_false_negative_prob)

                    prev_out_ind = prev_out_ind[random_subset_mask]
                    prev_target_ind = prev_target_ind[random_subset_mask]

                # detected prev frame tracks
                prev_track_ids = target['prev_track_ids'][prev_target_ind]

                # match track ids between frames
                target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
                target_ind_matching = target_ind_match_matrix.any(dim=1)
                target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

                # current frame track ids detected in the prev frame
                # track_ids = target['track_ids'][target_ind_matched_idx]

                # index of prev frame detection in current frame box list
                target['track_query_match_ids'] = target_ind_matched_idx

                # random false positives
                prev_boxes_matched = prev_out['pred_boxes'][i, prev_out_ind[target_ind_matching]]

                not_prev_out_ind = torch.arange(prev_out['pred_boxes'].shape[1])
                not_prev_out_ind = [
                    ind.item()
                    for ind in not_prev_out_ind
                    if ind not in prev_out_ind]

                random_false_out_ind = []
                for prev_box_matched in prev_boxes_matched:

                    if random.uniform(0, 1) < self._track_query_false_positive_prob:
                        prev_boxes_unmatched = prev_out['pred_boxes'][i, not_prev_out_ind]

                        # only cxcy
                        # box_dists = prev_box_matched[:2].sub(prev_boxes_unmatched[:, :2]).abs()
                        # box_dists = box_dists.pow(2).sum(dim=-1).sqrt()
                        # box_weights = 1.0 / box_dists.add(1e-8)

                        prev_box_ious, _ = box_ops.box_iou(
                            box_ops.box_cxcywh_to_xyxy(prev_box_matched.unsqueeze(dim=0)),
                            box_ops.box_cxcywh_to_xyxy(prev_boxes_unmatched))
                        box_weights = prev_box_ious[0]

                        if box_weights.gt(0.0).any():
                            random_false_out_idx = not_prev_out_ind.pop(
                                torch.multinomial(box_weights.cpu(), 1).item())
                            random_false_out_ind.append(random_false_out_idx)

                prev_out_ind = torch.tensor(prev_out_ind.tolist() + random_false_out_ind).long()
                target_ind_matching = torch.tensor(
                    target_ind_matching.tolist() + [False, ] * len(random_false_out_ind)).bool()

                # matches indices with 1.0 and not matched -1.0
                track_queries_match_mask = torch.ones_like(target_ind_matching).float()
                track_queries_match_mask[~target_ind_matching] = -1.0

                # set prev frame info
                hs_embeds = prev_out['hs_embed'][i, prev_out_ind]
                if self._track_query_noise and not torch.isnan(hs_embeds.std()).any():
                    track_query_noise = torch.randn_like(hs_embeds) \
                        * hs_embeds.std(dim=1, keepdim=True)
                    hs_embeds = hs_embeds + track_query_noise * self._track_query_noise
                    # hs_embeds = track_query_noise * self._track_query_noise \
                    #     + hs_embeds * (1 - self._track_query_noise)
                target['track_query_hs_embeds'] = hs_embeds
                target['track_query_boxes'] = prev_out['pred_boxes'][i, prev_out_ind].detach()

                # add zeros for detection object queries
                device = track_queries_match_mask.device
                track_queries_match_mask = torch.tensor(
                    track_queries_match_mask.tolist() + [0, ] * self.num_queries)

                target['track_queries_match_mask'] = track_queries_match_mask.to(device)

        out, targets, features, memory, hs  = super().forward(samples, targets)

        return out, targets, features, memory, hs


# TODO: with meta classes
class DETRTracking(DETRTrackingBase, DETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)


class DeformableDETRTracking(DETRTrackingBase, DeformableDETR):
    def __init__(self, tracking_kwargs, detr_kwargs):
        DeformableDETR.__init__(self, **detr_kwargs)
        DETRTrackingBase.__init__(self, **tracking_kwargs)
