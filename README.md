# TrackFormer: Multi-Object Tracking with Transformers

This repository provides the official implementation of the [TrackFormer: Multi-Object Tracking with Transformers](https://arxiv.org/abs/2101.02702) paper by [Tim Meinhardt](https://dvl.in.tum.de/team/meinhardt/), [Alexander Kirillov](https://alexander-kirillov.github.io/), [Laura Leal-Taixe](https://dvl.in.tum.de/team/lealtaixe/) and [Christoph Feichtenhofer](https://feichtenhofer.github.io/). The codebase builds upon [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw).

<!-- **As the paper is still under submission this repository will continuously be updated and might at times not reflect the current state of the [arXiv paper](https://arxiv.org/abs/2012.01866).** -->

<div align="center">
    <img src="docs/MOT17-03-SDP.gif" alt="MOT17-03-SDP" width="375"/>
    <img src="docs/MOTS20-07.gif" alt="MOTS20-07" width="375"/>
</div>

## Abstract

The challenging task of multi-object tracking (MOT) requires simultaneous reasoning about track initialization, identity, and spatiotemporal trajectories.
We formulate this task as a frame-to-frame set prediction problem and introduce TrackFormer, an end-to-end MOT approach based on an encoder-decoder Transformer architecture.
Our model achieves data association between frames via attention by evolving a set of track predictions through a video sequence.
The Transformer decoder initializes new tracks from static object queries and autoregressively follows existing tracks in space and time with the new concept of identity preserving track queries.
Both decoder query types benefit from self- and encoder-decoder attention on global frame-level features, thereby omitting any additional graph optimization and matching or modeling of motion and appearance.
TrackFormer represents a new tracking-by-attention paradigm and yields state-of-the-art performance on the task of multi-object tracking (MOT17) and segmentation (MOTS20).

<div align="center">
    <img src="docs/method.png" alt="TrackFormer casts multi-object tracking as a set prediction problem performing joint detection and tracking-by-attention. The architecture consists of a CNN for image feature extraction, a Transformer encoder for image feature encoding and a Transformer decoder which applies self- and encoder-decoder attention to produce output embeddings with bounding box and class information."/>
</div>

## Installation

We refer to our [docs/INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

## Train TrackFormer

We refer to our [docs/TRAIN.md](docs/TRAIN.md) for detailed training instructions.

## Evaluate TrackFormer

In order to evaluate TrackFormer on a multi-object tracking dataset, we provide the `src/track.py` script which supports several datasets and splits interchangle via the `dataset_name` argument (See `src/datasets/tracking/factory.py` for an overview of all datasets.) The default tracking configuration is specified in `cfgs/track.yaml`. To facilitate the reproducibility of our results, we provide evaluation metrics for both the train and test set.

### MOT17

#### Private detections

```
python src/track.py with reid
```

<center>

| MOT17     | MOTA         | IDF1           |       MT     |     ML     |     FP       |     FN              |  ID SW.      |
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  :---:       |
| **Train** |     74.2     |     71.7       |     849      | 177        |      7431    |      78057          |  1449        |
| **Test**  |     74.1     |     68.0       |    1113      | 246        |     34602    |     108777          |  2829        |

</center>

#### Public detections (DPM, FRCNN, SDP)

```
python src/track.py with \
    reid \
    tracker_cfg.public_detections=min_iou_0_5 \
    obj_detect_checkpoint_file=models/mot17_deformable_multi_frame/checkpoint_epoch_50.pth
```

<center>

| MOT17     | MOTA         | IDF1           |       MT     |     ML     |     FP       |     FN              |  ID SW.      |
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  :---:       |
| **Train** |     64.6     |     63.7       |    621       | 675        |     4827     |     111958          |  2556        |
| **Test**  |     62.3     |     57.6       |    688       | 638        |     16591    |     192123          |  4018        |

</center>

### MOT20

#### Private detections

```
python src/track.py with \
    reid \
    dataset_name=MOT20-ALL \
    obj_detect_checkpoint_file=models/mot20_crowdhuman_deformable_multi_frame/checkpoint_epoch_50.pth
```

<center>

| MOT20     | MOTA         | IDF1           |       MT     |     ML     |     FP       |     FN              |  ID SW.      |
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  :---:       |
| **Train** |     81.0     |     73.3       |    1540      | 124        |     20807    |     192665          |  1961        |
| **Test**  |     68.6     |     65.7       |     666      | 181        |     20348    |     140373          |  1532        |

</center>

### MOTS20

```
python src/track.py with \
    dataset_name=MOTS20-ALL \
    obj_detect_checkpoint_file=models/mots20_train_masks/checkpoint.pth
```

Our tracking script only applies MOT17 metrics evaluation but outputs MOTS20 mask prediction files. To evaluate these download the official [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit).

<center>

| MOTS20    | sMOTSA         | IDF1           |       FP     |     FN     |     IDs      |
|  :---:    | :---:          |     :---:      |    :---:     | :---:      |    :---:     |
| **Train** |     --         |     --         |    --        |   --       |     --       |
| **Test**  |     54.9       |     63.6       |    2233      | 7195       |     278      |

</center>

### Demo

To facilitate the application of TrackFormer, we provide a demo interface which allows for a quick processing of a given video sequence.

```
ffmpeg -i data/snakeboard/snakeboard.mp4 -vf fps=30 data/snakeboard/%06d.png

python src/track.py with \
    dataset_name=DEMO \
    data_root_dir=data/snakeboard \
    output_dir=data/snakeboard \
    write_images=pretty
```

<div align="center">
    <img src="docs/snakeboard.gif" alt="Snakeboard demo" width="600"/>
</div>

## Publication
If you use this software in your research, please cite our publication:

```
@InProceedings{meinhardt2021trackformer,
    title={TrackFormer: Multi-Object Tracking with Transformers},
    author={Tim Meinhardt and Alexander Kirillov and Laura Leal-Taixe and Christoph Feichtenhofer},
    year={2022},
    month = {June},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```