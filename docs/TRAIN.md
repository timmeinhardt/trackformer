# Train TrackFormer

We provide the code as well as intermediate models of our entire training pipeline for multiple datasets. Monitoring of the training/evaluation progress is possible via command line as well as [Visdom](https://github.com/fossasia/visdom.git). For the latter, a Visdom server must be running at `vis_port` and `vis_server` (see `cfgs/train.yaml`). We set `vis_server=''` by default to deactivate Visdom logging. To deactivate Visdom logging with set parameters, you can run a training with the `no_vis=True` flag.

<div align="center">
    <img src="../docs/visdom.gif" alt="Snakeboard demo" width="600"/>
</div>

The settings for each dataset are specified in the respective configuration files, e.g., `cfgs/train_crowdhuman.yaml`. The following train commands produced the pretrained model files mentioned in [docs/INSTALL.md](INSTALL.md).

## CrowdHuman pre-training

```
python src/train.py with \
    crowdhuman \
    deformable \
    multi_frame \
    tracking \
    output_dir=models/crowdhuman_deformable_multi_frame \
```

## MOT17

#### Private detections

```
python src/train.py with \
    mot17_crowdhuman \
    deformable \
    multi_frame \
    tracking \
    output_dir=models/mot17_crowdhuman_deformable_multi_frame \
```

#### Public detections

```
python src/train.py with \
    mot17 \
    deformable \
    multi_frame \
    tracking \
    output_dir=models/mot17_deformable_multi_frame \
```

## MOT20

#### Private detections

```
python src/train.py with \
    mot20_crowdhuman \
    deformable \
    multi_frame \
    tracking \
    output_dir=models/mot20_crowdhuman_deformable_multi_frame \
```

## MOTS20

For our MOTS20 test set submission, we finetune a MOT17 private detection model without deformable attention, i.e., vanilla DETR, which was pre-trained on the CrowdHuman dataset. The finetuning itself conists of two training steps: (i) the original DETR panoptic segmentation head on the COCO person segmentation data and (ii) the entire TrackFormer model (including segmentation head) on the MOTS20 training set. At this point, we only provide the final model files in [docs/INSTALL.md](INSTALL.md).

<!-- ```
python src/train.py with \
    tracking \
    coco_person_masks \
    output_dir=models/mot17_train_private_coco_person_masks_v2 \
```

```
python src/train.py with \
    tracking \
    mots20 \
    output_dir=models/mots20_train_masks \
``` -->

<!-- ### Ablation studies

Will be added after acceptance of the paper. -->

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
    mot17 \
    deformable \
    multi_frame \
    tracking \
    resume=models/mot17_crowdhuman_deformable_trackformer/checkpoint_epoch_40.pth \
    output_dir=models/custom_dataset_deformable \
    mot_path_train=data/custom_dataset \
    mot_path_val=data/custom_dataset \
    train_split=train \
    val_split=val \
    epochs=20 \
```

## Run with multipe GPUs

All reported results are obtained by training with a batch size of 2 and 7 GPUs, i.e., an effective batch size of 14. If you have less GPUs at your disposal, adjust the learning rates accordingly. To start the CrowdHuman pre-training with 7 GPUs execute:

```
python -m torch.distributed.launch --nproc_per_node=7 --use_env src/train.py with \
    crowdhuman \
    deformable \
    multi_frame \
    tracking \
    output_dir=models/crowdhuman_deformable_multi_frame \
```

## Run SLURM jobs with Submitit

Furthermore, we provide a script for starting Slurm jobs with [submitit](https://github.com/facebookincubator/submitit). This includes a convenient command line interface for Slurm options as well as preemption and resuming capabilities. The aforementioned CrowdHuman pre-training can be executed on 7 x 32 GB GPUs with the following command:

```
python src/run_with_submitit.py with \
    num_gpus=7 \
    vram=32GB \
    cluster=slurm \
    train.crowdhuman \
    train.deformable \
    train.trackformer \
    train.tracking \
    train.output_dir=models/crowdhuman_train_val_deformable \
```