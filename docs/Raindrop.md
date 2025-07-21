# Raindrop

[Official website](https://github.com/rui1996/DeRaindrop) | [Papers with Code](https://paperswithcode.com/dataset/raindrop) | [Google drive](https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K)

## Brief introduction

> Copied from paperswithcode.

Raindrop is a set of image pairs, where each pair contains exactly the same background scene, yet one is degraded by raindrops and the other one is free from raindrops. To obtain this, the images are captured through two pieces of exactly the same glass: one sprayed with water, and the other is left clean. The dataset consists of 1,119 pairs of images, with various background scenes and raindrops. They were captured with a Sony A6000 and a Canon EOS 60.

## Statistics

**Numbers**: 1,110

**Splits** (train / test_a / test_b): 861 / 58 / 249

**Resolution**: ~720x480

**Note**: test_a is a subset of test_b where the alignment of image pairs is good.

## Usage

### Download

```shell
gdown 1fQhdGTGeUjH4FoG3bD-W_LiRq0gOhLeu -O train.zip
gdown 1a4ZtiK_Sfuowkc19NjTvlA9on8HEymLF -O test_a.zip
gdown 1guI1E8dVQ2RwY0KY9Htf_R-MM8jTsaha -O test_b.zip
```

### File structure

Please organize the downloaded dataset in the following file structure:

```text
root
├── train
│   ├── data
│   │   ├── 0_rain.png
│   │   ├── ...
│   │   └── 860_rain.png
│   ├── gt
│   │   ├── 0_clean.png
│   │   ├── ...
│   │   └── 860_clean.png
│   └── preview.html
├── test_a
│   ├── data
│   │   ├── 0_rain.png
│   │   ├── ...
│   │   └── 57_rain.png
│   └── gt
│       ├── 0_clean.png
│       ├── ...
│       └── 57_clean.png
└── test_b
    ├── data
    │   ├── 0_rain.jpg
    │   ├── ...
    │   └── 248_rain.jpg
    └── gt
        ├── 0_clean.jpg
        ├── ...
        └── 248_clean.jpg
```

### Example

```python
from image_datasets import Raindrop

root = '~/data/Raindrop'  # path to downloaded dataset
train_set = Raindrop(root=root, split='train')
testa_set = Raindrop(root=root, split='test_a')
testb_set = Raindrop(root=root, split='test_b')
print(len(train_set))  # 861
print(len(testa_set))  # 58
print(len(testb_set))  # 249
print(train_set[0]['image'].shape)  # (3, 480, 720)
print(train_set[0]['gt'].shape)     # (3, 480, 720)
```
