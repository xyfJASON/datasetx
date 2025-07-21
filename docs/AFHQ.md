# AFHQ

[Official website](https://github.com/clovaai/stargan-v2) | [Papers with Code](https://paperswithcode.com/dataset/afhq) | [Dropbox](https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0)

## Brief introduction

> Copied from paperswithcode.

Animal FacesHQ (AFHQ) is a dataset of animal faces consisting of 15,000 high-quality images at 512 × 512 resolution. The dataset includes three domains of cat, dog, and wildlife, each providing 5000 images. By having multiple (three) domains and diverse images of various breeds (≥ eight) per each domain, AFHQ sets a more challenging image-to-image translation problem. All images are vertically and horizontally aligned to have the eyes at the center. The low-quality images were discarded by human effort.

## Statistics

> The stats below are for AFHQv2, an upgrade from AFHQv1 that used a better resampling method (nearest neighbor -> lanczos), removed about 2% of the images, and saved in PNG format.

**Numbers**: 15,803

**Splits** (train / test):
- Total: 14,336 / 1,467
- Cat: 5,065 / 493
- Dog: 4,678 / 491
- Wild: 4,593 / 483

**Resolution**: 512×512

## Usage

### Download

```shell
wget -N https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0
unzip afhq_v2.zip
```

### File structure

```text
root
├── afhq_v2.zip    (7.0 GB)
├── train          (extracted from afhq_v2.zip)
│   ├── cat        (contains 5065 images)
│   ├── dog        (contains 4678 images)
│   └── wild       (contains 4593 images)
└── test           (extracted from afhq_v2.zip)
    ├── cat        (contains 493 images)
    ├── dog        (contains 491 images)
    └── wild       (contains 483 images)
```

### Example

```python
from image_datasets import AFHQ

root = '~/data/AFHQ'   # path to downloaded dataset
train_set = AFHQ(root=root, split='train')
test_set = AFHQ(root=root, split='test')
print(len(train_set))  # 14336
print(len(test_set))   # 1467
print(train_set[0]['image'].shape)  # (3, 512, 512)
print(train_set[0]['label'])  # 0
```
