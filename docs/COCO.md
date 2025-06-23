# COCO

[Official website](https://cocodataset.org) | [Papers with Code](https://paperswithcode.com/dataset/coco) | [Guide](https://www.v7labs.com/blog/coco-dataset-guide)

## Brief introduction

> Copied from paperswithcode.

The MS COCO (Microsoft Common Objects in Context) dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images.

## Statistics (COCO2017)

**Numbers**: 163,957

**Splits** (train / valid / test / unlabeled): 118,287 / 5,000 / 40,670 / 123,403

**Resolution**: Mostly around 640x480

**Annotations** (copied from paperswithcode):

 - object detection: bounding boxes and per-instance segmentation masks with 80 object categories,
 - captioning: natural language descriptions of the images (see MS COCO Captions),
 - keypoints detection: containing more than 200,000 images and 250,000 person instances labeled with keypoints (17 possible keypoints, such as left eye, nose, right hip, right ankle),
 - stuff image segmentation: per-pixel segmentation masks with 91 stuff categories, such as grass, wall, sky (see MS COCO Stuff),
 - panoptic: full scene segmentation, with 80 thing categories (such as person, bicycle, elephant) and a subset of 91 stuff categories (grass, sky, road),
 - dense pose: more than 39,000 images and 56,000 person instances labeled with DensePose annotations – each labeled person is annotated with an instance id and a mapping between image pixels that belong to that person body and a template 3D model. The annotations are publicly available only for training and validation images.

## Usage

### File structure

Please organize the downloaded dataset in the following file structure:

```text
root
├── annotations
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── image_info_test2017.json
│   ├── image_info_test-dev2017.json
│   ├── image_info_unlabeled2017.json
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── panoptic_train2017.json
│   ├── panoptic_train2017
│   │   ├── 000000000009.png
│   │   ├── ...
│   │   └── 000000581929.png
│   ├── panoptic_val2017.json
│   ├── panoptic_val2017
│   │   ├── 000000000139.png
│   │   ├── ...
│   │   └── 000000581781.png
│   ├── person_keypoints_train2017.json
│   ├── person_keypoints_val2017.json
│   ├── stuff_train2017.json
│   ├── stuff_train2017_pixelmaps
│   │   ├── 000000000009.png
│   │   ├── ...
│   │   └── 000000581929.png
│   ├── stuff_val2017.json
│   └── stuff_val2017_pixelmaps
│       ├── 000000000139.png
│       ├── ...
│       └── 000000581781.png
├── train2017
│   ├── 000000000009.jpg
│   ├── ...
│   └── 000000581929.jpg
├── val2017
│   ├── 000000000139.jpg
│   ├── ...
│   └── 000000581781.jpg
├── test2017
│   ├── 000000000001.jpg
│   ├── ...
│   └── 000000581918.jpg
└── unlabeled2017
    ├── 000000000008.jpg
    ├── ...
    └── 000000581931.jpg
```
