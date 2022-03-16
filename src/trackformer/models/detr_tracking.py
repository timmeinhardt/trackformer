import math
import random
from contextlib import nullcontext

import torch
import torch.nn as nn

from ..util import box_ops
from ..util.misc import NestedTensor, get_rank
from .deformable_detr import DeformableDETR
from .detr import DETR
from .matcher import HungarianMatcher


class DETRTrackingBase(nn.Module):

    def __init__(self,
                 track_query_false_positive_prob: float = 0.0,
                 track_query_false_negative_prob: float = 0.0,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False):
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._backprop_prev_frame = backprop_prev_frame

        self._tracking = False

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True

    def add_track_queries_to_targets(self, targets, prev_indices, prev_out, add_false_pos=True):
        device = prev_out['pred_boxes'].device

        # for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
        min_prev_target_ind = min([len(prev_ind[1]) for prev_ind in prev_indices])
        num_prev_target_ind = 0
        if min_prev_target_ind:
            num_prev_target_ind = torch.randint(0, min_prev_target_ind + 1, (1,)).item()

        num_prev_target_ind_for_fps = 0
        if num_prev_target_ind:
            num_prev_target_ind_for_fps = \
                torch.randint(int(math.ceil(self._track_query_false_positive_prob * num_prev_target_ind)) + 1, (1,)).item()

        for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
            prev_out_ind, prev_target_ind = prev_ind

            # random subset
            if self._track_query_false_negative_prob: # and len(prev_target_ind):
                # random_subset_mask = torch.empty(len(prev_target_ind)).uniform_()
                # random_subset_mask = random_subset_mask.ge(
                #     self._track_query_false_negative_prob)

                # random_subset_mask = torch.randperm(len(prev_target_ind))[:torch.randint(0, len(prev_target_ind) + 1, (1,))]
                random_subset_mask = torch.randperm(len(prev_target_ind))[:num_prev_target_ind]

                # if not len(random_subset_mask):
                #     target['track_query_hs_embeds'] = torch.zeros(0, self.hidden_dim).float().to(device)
                #     target['track_queries_placeholder_mask'] = torch.zeros(self.num_queries).bool().to(device)
                #     target['track_queries_mask'] = torch.zeros(self.num_queries).bool().to(device)
                #     target['track_queries_fal_pos_mask'] = torch.zeros(self.num_queries).bool().to(device)
                #     target['track_query_boxes'] = torch.zeros(0, 4).to(device)
                #     target['track_query_match_ids'] = torch.tensor([]).long().to(device)

                #     continue

                prev_out_ind = prev_out_ind[random_subset_mask]
                prev_target_ind = prev_target_ind[random_subset_mask]

            # detected prev frame tracks
            prev_track_ids = target['prev_target']['track_ids'][prev_target_ind]

            # match track ids between frames
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
            target_ind_matching = target_ind_match_matrix.any(dim=1)
            target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

            # current frame track ids detected in the prev frame
            # track_ids = target['track_ids'][target_ind_matched_idx]

            # index of prev frame detection in current frame box list
            target['track_query_match_ids'] = target_ind_matched_idx

            # random false positives
            if add_false_pos:
                prev_boxes_matched = prev_out['pred_boxes'][i, prev_out_ind[target_ind_matching]]

                not_prev_out_ind = torch.arange(prev_out['pred_boxes'].shape[1])
                not_prev_out_ind = [
                    ind.item()
                    for ind in not_prev_out_ind
                    if ind not in prev_out_ind]

                random_false_out_ind = []

                prev_target_ind_for_fps = torch.randperm(num_prev_target_ind)[:num_prev_target_ind_for_fps]

                # for j, prev_box_matched in enumerate(prev_boxes_matched):
                #     if j not in prev_target_ind_for_fps:
                #         continue

                for j in prev_target_ind_for_fps:
                    # if random.uniform(0, 1) < self._track_query_false_positive_prob:
                    prev_boxes_unmatched = prev_out['pred_boxes'][i, not_prev_out_ind]

                    # only cxcy
                    # box_dists = prev_box_matched[:2].sub(prev_boxes_unmatched[:, :2]).abs()
                    # box_dists = box_dists.pow(2).sum(dim=-1).sqrt()
                    # box_weights = 1.0 / box_dists.add(1e-8)

                    # prev_box_ious, _ = box_ops.box_iou(
                    #     box_ops.box_cxcywh_to_xyxy(prev_box_matched.unsqueeze(dim=0)),
                    #     box_ops.box_cxcywh_to_xyxy(prev_boxes_unmatched))
                    # box_weights = prev_box_ious[0]

                    # dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

                    if len(prev_boxes_matched) > j:
                        prev_box_matched = prev_boxes_matched[j]
                        box_weights = \
                            prev_box_matched.unsqueeze(dim=0)[:, :2] - \
                            prev_boxes_unmatched[:, :2]
                        box_weights = box_weights[:, 0] ** 2 + box_weights[:, 0] ** 2
                        box_weights = torch.sqrt(box_weights)

                        # if box_weights.gt(0.0).any():
                        # if box_weights.gt(0.0).any():
                        random_false_out_idx = not_prev_out_ind.pop(
                            torch.multinomial(box_weights.cpu(), 1).item())
                    else:
                        random_false_out_idx = not_prev_out_ind.pop(torch.randperm(len(not_prev_out_ind))[0])

                    random_false_out_ind.append(random_false_out_idx)

                prev_out_ind = torch.tensor(prev_out_ind.tolist() + random_false_out_ind).long()

                target_ind_matching = torch.cat([
                    target_ind_matching,
                    torch.tensor([False, ] * len(random_false_out_ind)).bool().to(device)
                ])

            # MSDeformAttn can not deal with empty inputs therefore we
            # add single false pos to have at least one track query per sample
            # not_prev_out_ind = torch.tensor([
            #     ind
            #     for ind in torch.arange(prev_out['pred_boxes'].shape[1])
            #     if ind not in prev_out_ind])
            # false_samples_inds = torch.randperm(not_prev_out_ind.size(0))[:1]
            # false_samples = not_prev_out_ind[false_samples_inds]
            # prev_out_ind = torch.cat([prev_out_ind, false_samples])
            # target_ind_matching = torch.tensor(
            #     target_ind_matching.tolist() + [False, ]).bool().to(target_ind_matching.device)

            # track query masks
            track_queries_mask = torch.ones_like(target_ind_matching).bool()
            track_queries_fal_pos_mask = torch.zeros_like(target_ind_matching).bool()
            track_queries_fal_pos_mask[~target_ind_matching] = True

            # track_queries_match_mask = torch.ones_like(target_ind_matching).float()
            # matches indices with 1.0 and not matched -1.0
            # track_queries_mask[~target_ind_matching] = -1.0

            # set prev frame info
            target['track_query_hs_embeds'] = prev_out['hs_embed'][i, prev_out_ind]
            target['track_query_boxes'] = prev_out['pred_boxes'][i, prev_out_ind].detach()

            target['track_queries_mask'] = torch.cat([
                track_queries_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

            target['track_queries_fal_pos_mask'] = torch.cat([
                track_queries_fal_pos_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

        # add placeholder track queries to allow for batch sizes > 1
        # max_track_query_hs_embeds = max([len(t['track_query_hs_embeds']) for t in targets])
        # for i, target in enumerate(targets):

        #     num_add = max_track_query_hs_embeds - len(target['track_query_hs_embeds'])

        #     if not num_add:
        #         target['track_queries_placeholder_mask'] = torch.zeros_like(target['track_queries_mask']).bool()
        #         continue

        #     raise NotImplementedError

        #     target['track_query_hs_embeds'] = torch.cat(
        #         [torch.zeros(num_add, self.hidden_dim).to(device),
        #          target['track_query_hs_embeds']
        #     ])
        #     target['track_query_boxes'] = torch.cat(
        #         [torch.zeros(num_add, 4).to(device),
        #          target['track_query_boxes']
        #     ])

        #     target['track_queries_mask'] = torch.cat([
        #         torch.tensor([True, ] * num_add).to(device),
        #         target['track_queries_mask']
        #     ]).bool()

        #     target['track_queries_fal_pos_mask'] = torch.cat([
        #         torch.tensor([False, ] * num_add).to(device),
        #         target['track_queries_fal_pos_mask']
        #     ]).bool()

        #     target['track_queries_placeholder_mask'] = torch.zeros_like(target['track_queries_mask']).bool()
        #     target['track_queries_placeholder_mask'][:num_add] = True

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        if targets is not None and not self._tracking:
            prev_targets = [target['prev_target'] for target in targets]

            # if self.training and random.uniform(0, 1) < 0.5:
            if self.training:
            # if True:
                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext

                with backprop_context():
                    if 'prev_prev_image' in targets[0]:
                        for target, prev_target in zip(targets, prev_targets):
                            prev_target['prev_target'] = target['prev_prev_target']

                        prev_prev_targets = [target['prev_prev_target'] for target in targets]

                        # PREV PREV
                        prev_prev_out, _, prev_prev_features, _, _ = super().forward([t['prev_prev_image'] for t in targets])

                        prev_prev_outputs_without_aux = {
                            k: v for k, v in prev_prev_out.items() if 'aux_outputs' not in k}
                        prev_prev_indices = self._matcher(prev_prev_outputs_without_aux, prev_prev_targets)

                        self.add_track_queries_to_targets(
                            prev_targets, prev_prev_indices, prev_prev_out, add_false_pos=False)

                        # PREV
                        prev_out, _, prev_features, _, _ = super().forward(
                            [t['prev_image'] for t in targets],
                            prev_targets,
                            prev_prev_features)
                    else:
                        prev_out, _, prev_features, _, _ = super().forward([t['prev_image'] for t in targets])

                    # prev_out = {k: v.detach() for k, v in prev_out.items() if torch.is_tensor(v)}

                    prev_outputs_without_aux = {
                        k: v for k, v in prev_out.items() if 'aux_outputs' not in k}
                    prev_indices = self._matcher(prev_outputs_without_aux, prev_targets)

                    self.add_track_queries_to_targets(targets, prev_indices, prev_out)
            else:
                # if not training we do not add track queries and evaluate detection performance only.
                # tracking performance is evaluated by the actual tracking evaluation.
                for target in targets:
                    device = target['boxes'].device

                    target['track_query_hs_embeds'] = torch.zeros(0, self.hidden_dim).float().to(device)
                    # target['track_queries_placeholder_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_queries_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_queries_fal_pos_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_query_boxes'] = torch.zeros(0, 4).to(device)
                    target['track_query_match_ids'] = torch.tensor([]).long().to(device)

        out, targets, features, memory, hs  = super().forward(samples, targets, prev_features)

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
