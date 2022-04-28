# Installation

1. Clone and enter this repository:
    ```
    git clone git@github.com:timmeinhardt/trackformer.git
    cd trackformer
    ```

2. Install packages for Python 3.7:

    1. `pip3 install -r requirements.txt`
    2. Install PyTorch 1.5 and torchvision 0.6 from [here](https://pytorch.org/get-started/previous-versions/#v150).
    3. Install pycocotools (with fixed ignore flag): `pip3 install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'`
    5. Install MultiScaleDeformableAttention package: `python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install`

3. Download and unpack datasets in the `data` directory:

    1. [MOT17](https://motchallenge.net/data/MOT17/):

        ```
        wget https://motchallenge.net/data/MOT17.zip
        unzip MOT17.zip
        python src/generate_coco_from_mot.py
        ```

    2. (Optional) [MOT20](https://motchallenge.net/data/MOT20/):

        ```
        wget https://motchallenge.net/data/MOT20.zip
        unzip MOT20.zip
        python src/generate_coco_from_mot.py --mot20
        ```

    3. (Optional) [MOTS20](https://motchallenge.net/data/MOTS/):

        ```
        wget https://motchallenge.net/data/MOTS.zip
        unzip MOTS.zip
        python src/generate_coco_from_mot.py --mots
        ```

    4. (Optional) [CrowdHuman](https://www.crowdhuman.org/download.html):

        1. Create a `CrowdHuman` and `CrowdHuman/annotations` directory.
        2. Download and extract the `train` and `val` datasets including their corresponding `*.odgt` annotation file into the `CrowdHuman` directory.
        3. Create a `CrowdHuman/train_val` directory and merge or symlink the `train` and `val` image folders.
        4. Run `python src/generate_coco_from_crowdhuman.py`
        5. The final folder structure should resemble this:
            ~~~
            |-- data
                |-- CrowdHuman
                |   |-- train
                |   |   |-- *.jpg
                |   |-- val
                |   |   |-- *.jpg
                |   |-- train_val
                |   |   |-- *.jpg
                |   |-- annotations
                |   |   |-- annotation_train.odgt
                |   |   |-- annotation_val.odgt
                |   |   |-- train_val.json
            ~~~

3. Download and unpack pretrained TrackFormer model files in the `models` directory:

    ```
    wget https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip
    unzip trackformer_models_v1.zip
    ```

4. (optional) The evaluation of MOTS20 metrics requires two steps:
    1. Run Trackformer with `src/track.py` and output prediction files
    2. Download the official MOTChallenge [devkit](https://github.com/dendorferpatrick/MOTChallengeEvalKit) and run the MOTS evaluation on the prediction files

In order to configure, log and reproduce our computational experiments, we structure our code with the [Sacred](http://sacred.readthedocs.io/en/latest/index.html) framework. For a detailed explanation of the Sacred interface please read its documentation.
