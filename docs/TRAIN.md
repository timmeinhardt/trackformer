# Train TrackFormer

We provide the code as well as intermediate models of our entire training pipeline for multiple datasets. Monitoring of the training/evaluation progress is possible via command line as well as [Visdom](https://github.com/fossasia/visdom.git). For the latter, a Visdom server must be running at `vis_port=8090` and `vis_server=http://localhost` (see `cfgs/train.yaml`). To deactivate Visdom logging run a training with the `no_vis=True` flag.

<div align="center">
    <img src="../docs/visdom.gif" alt="Snakeboard demo" width="600"/>
</div>

The settings for each dataset are specified in the respective configuration files, e.g., `cfgs/train_crowdhuman.yaml`.

## CrowdHuman pre-training

```
python src/train.py with \
    deformable \
    tracking \
    crowdhuman \
    full_res \
    output_dir=models/crowdhuman_train_val_deformable_v2 \
```

## MOT17

#### Private detections

```
python src/train.py with \
    deformable \
    tracking \
    mot17 \
    full_res \
    resume=models/crowdhuman_train_val_deformable/checkpoint.pth \
    output_dir=models/mot17_train_deformable_private_v2 \
```

#### Public detections

```
python src/train.py with \
    deformable \
    tracking \
    mot17 \
    full_res \
    resume=models/r50_deformable_detr-checkpoint.pth \
    output_dir=models/mot17_train_deformable_public_v2 \
    epochs=40 \
    lr_drop=10
```

## MOTS20

For our MOTS20 test set submission, we finetune a MOT17 private detection model without deformable attention, i.e., vanilla DETR, which was pre-trained on the CrowdHuman dataset. The finetuning itself conists of two training steps: (i) the original DETR panoptic segmentation head on the COCO person segmentation data and (ii) the entire TrackFormer model (including segmentation head) on the MOTS20 training set.

```
python src/train.py with \
    tracking \
    coco_person_masks \
    output_dir=models/mot17_train_private_coco_person_masks_v2 \
```

```
python src/train.py with \
    tracking \
    mots20 \
    output_dir=models/mots20_train_masks_v2 \
```

### Ablation studies

Will be added after acceptance of the paper.

## Custom Dataset

TrackFormer can be trained on additional/new object detection or multi-object tracking datasets without changing our codebase. The `crowdhuman` or `mot` datasets merely require a [COCO style](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch) annotation file and the following folder structure:

~~~
|-- data
    |-- custom_dataset
    |   |-- train
    |   |   |-- *.jpg
    |   |-- val
    |   |   |-- *.jpg
    |   |-- annotations
    |   |   |-- train.json
    |   |   |-- val.json
~~~

In the case of a multi-object tracking dataset, the original COCO annotations style must be extended with `seq_length`, `first_frame_image_id` and `track_id` fields. See the `src/generate_coco_from_mot.py` script for details. For example, the following command finetunes our `MOT17` private model for additional 20 epochs on a custom dataset:

```
python src/train.py with \
    deformable \
    tracking \
    mot17 \
    full_res \
    resume=models/mot17_train_deformable_private/checkpoint.pth \
    output_dir=models/custom_dataset_train_deformable \
    mot_path=data/custom_dataset \
    train_split=train \
    val_split=val \
    epochs=20 \
```

## Run with Submitit

Furthermore, we provide a script for starting Slurm jobs with [submitit](https://github.com/facebookincubator/submitit). This includes a convenient command line interface for Slurm options as well as preemption and resuming capabilities. The aforementioned CrowdHuman pre-training can be executed on 8 x 16 GB GPUs with the following command:

```
python src/run_with_submitit.py with \
    num_gpus=8 \
    vram=16GB \
    cluster=slurm \
    train.deformable \
    train.tracking \
    train.crowdhuman \
    train.full_res \
    train.output_dir=models/crowdhuman_train_val_deformable_v2 \
```