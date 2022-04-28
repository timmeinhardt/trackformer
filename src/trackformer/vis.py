import copy
import logging

import matplotlib.patches as mpatches
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib import colors
from matplotlib import pyplot as plt
from torchvision.ops.boxes import clip_boxes_to_image
from visdom import Visdom

from .util.plot_utils import fig_to_numpy

logging.getLogger('visdom').setLevel(logging.CRITICAL)


class BaseVis(object):

    def __init__(self, viz_opts, update_mode='append', env=None, win=None,
                 resume=False, port=8097, server='http://localhost'):
        self.viz_opts = viz_opts
        self.update_mode = update_mode
        self.win = win
        if env is None:
            env = 'main'
        self.viz = Visdom(env=env, port=port, server=server)
        # if resume first plot should not update with replace
        self.removed = not resume

    def win_exists(self):
        return self.viz.win_exists(self.win)

    def close(self):
        if self.win is not None:
            self.viz.close(win=self.win)
            self.win = None

    def register_event_handler(self, handler):
        self.viz.register_event_handler(handler, self.win)


class LineVis(BaseVis):
    """Visdom Line Visualization Helper Class."""

    def plot(self, y_data, x_label):
        """Plot given data.

        Appends new data to exisiting line visualization.
        """
        update = self.update_mode
        # update mode must be None the first time or after plot data was removed
        if self.removed:
            update = None
            self.removed = False

        if isinstance(x_label, list):
            Y = torch.Tensor(y_data)
            X = torch.Tensor(x_label)
        else:
            y_data = [d.cpu() if torch.is_tensor(d)
                      else torch.tensor(d)
                      for d in y_data]

            Y = torch.Tensor(y_data).unsqueeze(dim=0)
            X = torch.Tensor([x_label])

        win = self.viz.line(X=X, Y=Y, opts=self.viz_opts, win=self.win, update=update)

        if self.win is None:
            self.win = win
        self.viz.save([self.viz.env])

    def reset(self):
        #TODO: currently reset does not empty directly only on the next plot.
        # update='remove' is not working as expected.
        if self.win is not None:
            # self.viz.line(X=None, Y=None, win=self.win, update='remove')
            self.removed = True


class ImgVis(BaseVis):
    """Visdom Image Visualization Helper Class."""

    def plot(self, images):
        """Plot given images."""

        # images = [img.data if isinstance(img, torch.autograd.Variable)
        #           else img for img in images]
        # images = [img.squeeze(dim=0) if len(img.size()) == 4
        #           else img for img in images]

        self.win = self.viz.images(
            images,
            nrow=1,
            opts=self.viz_opts,
            win=self.win, )
        self.viz.save([self.viz.env])


def vis_results(visualizer, img, result, target, tracking):
    inv_normalize = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )

    imgs = [inv_normalize(img).cpu()]
    img_ids = [target['image_id'].item()]
    for key in ['prev', 'prev_prev']:
        if f'{key}_image' in target:
            imgs.append(inv_normalize(target[f'{key}_image']).cpu())
            img_ids.append(target[f'{key}_target'][f'image_id'].item())

    # img.shape=[3, H, W]
    dpi = 96
    figure, axarr = plt.subplots(len(imgs))
    figure.tight_layout()
    figure.set_dpi(dpi)
    figure.set_size_inches(
        imgs[0].shape[2] / dpi,
        imgs[0].shape[1] * len(imgs) / dpi)

    if len(imgs) == 1:
        axarr = [axarr]

    for ax, img, img_id in zip(axarr, imgs, img_ids):
        ax.set_axis_off()
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1))

        ax.text(
            0, 0, f'IMG_ID={img_id}',
            fontsize=20, bbox=dict(facecolor='white', alpha=0.5))

    num_track_queries = num_track_queries_with_id = 0
    if tracking:
        num_track_queries = len(target['track_query_boxes'])
        num_track_queries_with_id = len(target['track_query_match_ids'])
        track_ids = target['track_ids'][target['track_query_match_ids']]

    keep = result['scores'].cpu() > result['scores_no_object'].cpu()

    cmap = plt.cm.get_cmap('hsv', len(keep))

    prop_i = 0
    for box_id in range(len(keep)):
        rect_color = 'green'
        offset = 0
        text = f"{result['scores'][box_id]:0.2f}"

        if tracking:
            if target['track_queries_fal_pos_mask'][box_id]:
                rect_color = 'red'
            elif target['track_queries_mask'][box_id]:
                offset = 50
                rect_color = 'blue'
                text = (
                    f"{track_ids[prop_i]}\n"
                    f"{text}\n"
                    f"{result['track_queries_with_id_iou'][prop_i]:0.2f}")
                prop_i += 1

        if not keep[box_id]:
            continue

        # x1, y1, x2, y2 = result['boxes'][box_id]
        result_boxes = clip_boxes_to_image(result['boxes'], target['size'])
        x1, y1, x2, y2 = result_boxes[box_id]

        axarr[0].add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color=rect_color, linewidth=2))

        axarr[0].text(
            x1, y1 + offset, text,
            fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        if 'masks' in result:
            mask = result['masks'][box_id][0].numpy()
            mask = np.ma.masked_where(mask == 0.0, mask)

            axarr[0].imshow(
                mask, alpha=0.5, cmap=colors.ListedColormap([cmap(box_id)]))

    query_keep = keep
    if tracking:
        query_keep = keep[target['track_queries_mask'] == 0]

    legend_handles = [mpatches.Patch(
        color='green',
        label=f"object queries ({query_keep.sum()}/{len(target['boxes']) - num_track_queries_with_id})\n- cls_score")]

    if num_track_queries:
        track_queries_label = (
            f"track queries ({keep[target['track_queries_mask']].sum() - keep[target['track_queries_fal_pos_mask']].sum()}"
            f"/{num_track_queries_with_id})\n- track_id\n- cls_score\n- iou")

        legend_handles.append(mpatches.Patch(
            color='blue',
            label=track_queries_label))

    if num_track_queries_with_id != num_track_queries:
        track_queries_fal_pos_label = (
            f"false track queries ({keep[target['track_queries_fal_pos_mask']].sum()}"
            f"/{num_track_queries - num_track_queries_with_id})")

        legend_handles.append(mpatches.Patch(
            color='red',
            label=track_queries_fal_pos_label))

    axarr[0].legend(handles=legend_handles)

    i = 1
    for frame_prefix in ['prev', 'prev_prev']:
        # if f'{frame_prefix}_image_id' not in target or f'{frame_prefix}_boxes' not in target:
        if f'{frame_prefix}_target' not in target:
            continue

        frame_target = target[f'{frame_prefix}_target']
        cmap = plt.cm.get_cmap('hsv', len(frame_target['track_ids']))

        for j, track_id in enumerate(frame_target['track_ids']):
            x1, y1, x2, y2 = frame_target['boxes'][j]
            axarr[i].text(
                x1, y1, f"track_id={track_id}",
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            axarr[i].add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, color='green', linewidth=2))

            if 'masks' in frame_target:
                mask = frame_target['masks'][j].cpu().numpy()
                mask = np.ma.masked_where(mask == 0.0, mask)

                axarr[i].imshow(
                    mask, alpha=0.5, cmap=colors.ListedColormap([cmap(j)]))
        i += 1

    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.axis('off')

    img = fig_to_numpy(figure).transpose(2, 0, 1)
    plt.close()

    visualizer.plot(img)


def build_visualizers(args: dict, train_loss_names: list):
    visualizers = {}
    visualizers['train'] = {}
    visualizers['val'] = {}

    if args.eval_only or args.no_vis or not args.vis_server:
        return visualizers

    env_name = str(args.output_dir).split('/')[-1]

    vis_kwargs = {
        'env': env_name,
        'resume': args.resume and args.resume_vis,
        'port': args.vis_port,
        'server': args.vis_server}

    #
    # METRICS
    #

    legend = ['loss']
    legend.extend(train_loss_names)
    # for i in range(len(train_loss_names)):
    #     legend.append(f"{train_loss_names[i]}_unscaled")

    legend.extend([
        'class_error',
        # 'loss',
        # 'loss_bbox',
        # 'loss_ce',
        # 'loss_giou',
        # 'loss_mask',
        # 'loss_dice',
        # 'cardinality_error_unscaled',
        # 'loss_bbox_unscaled',
        # 'loss_ce_unscaled',
        # 'loss_giou_unscaled',
        # 'loss_mask_unscaled',
        # 'loss_dice_unscaled',
        'lr',
        'lr_backbone',
        'iter_time'
    ])

    # if not args.masks:
    #     legend.remove('loss_mask')
    #     legend.remove('loss_mask_unscaled')
    #     legend.remove('loss_dice')
    #     legend.remove('loss_dice_unscaled')

    opts = dict(
        title="TRAIN METRICS ITERS",
        xlabel='ITERS',
        ylabel='METRICS',
        width=1000,
        height=500,
        legend=legend)

    # TRAIN
    visualizers['train']['iter_metrics'] = LineVis(opts, **vis_kwargs)

    opts = copy.deepcopy(opts)
    opts['title'] = "TRAIN METRICS EPOCHS"
    opts['xlabel'] = "EPOCHS"
    opts['legend'].remove('lr')
    opts['legend'].remove('lr_backbone')
    opts['legend'].remove('iter_time')
    visualizers['train']['epoch_metrics'] = LineVis(opts, **vis_kwargs)

    # VAL
    opts = copy.deepcopy(opts)
    opts['title'] = "VAL METRICS EPOCHS"
    opts['xlabel'] = "EPOCHS"
    visualizers['val']['epoch_metrics'] = LineVis(opts, **vis_kwargs)

    #
    # EVAL COCO
    #

    legend = [
        'BBOX AP IoU=0.50:0.95',
        'BBOX AP IoU=0.50',
        'BBOX AP IoU=0.75',
    ]

    if args.masks:
        legend.extend([
            'MASK AP IoU=0.50:0.95',
            'MASK AP IoU=0.50',
            'MASK AP IoU=0.75'])

    if args.tracking and args.tracking_eval:
        legend.extend(['MOTA', 'IDF1'])

    opts = dict(
        title='TRAIN EVAL EPOCHS',
        xlabel='EPOCHS',
        ylabel='METRICS',
        width=1000,
        height=500,
        legend=legend)

    # TRAIN
    visualizers['train']['epoch_eval'] = LineVis(opts, **vis_kwargs)

    # VAL
    opts = copy.deepcopy(opts)
    opts['title'] = 'VAL EVAL EPOCHS'
    visualizers['val']['epoch_eval'] = LineVis(opts, **vis_kwargs)

    #
    # EXAMPLE RESULTS
    #

    opts = dict(
        title="TRAIN EXAMPLE RESULTS",
        width=2500,
        height=2500)

    # TRAIN
    visualizers['train']['example_results'] = ImgVis(opts, **vis_kwargs)

    # VAL
    opts = copy.deepcopy(opts)
    opts['title'] = 'VAL EXAMPLE RESULTS'
    visualizers['val']['example_results'] = ImgVis(opts, **vis_kwargs)

    return visualizers
