# CelebA-HQ

[Official website](https://github.com/tkarras/progressive_growing_of_gans) | [Papers with Code](https://paperswithcode.com/dataset/celeba-hq)

## Brief introduction

> Copied from paperswithcode.

The CelebA-HQ dataset is a high-quality version of CelebA that consists of 30,000 images at 1024×1024 resolution.

## Statistics

**Numbers**: 30,000 (a subset of CelebA)

**Splits** (train / valid / test): 24,183 / 2,993 / 2,824 (following CelebA's original splits)

**Resolution**: 1024×1024

## Usage

### Generate the dataset (official)

Download CelebA dataset and `delta` files, then generate images with `dataset_tool.py`. See [official repo](https://github.com/tkarras/progressive_growing_of_gans) for more information.

### Generate the dataset (recommended)

Download [CelebAMask-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html) dataset, then map the filenames back to original id based on `CelebA-HQ-to-CelebA-mapping.txt`. The mapping script is provided at [`scripts/celebahq_map_filenames.py`](../scripts/celebahq_map_filenames.py).

```shell
python celebahq_map_filenames.py --root ROOT
```

### File structure

Please organize the dataset in the following file structure:

```text
root
├── CelebA-HQ-img
│   ├── 000004.jpg
│   ├── ...
│   └── 202591.jpg
└── CelebA-HQ-to-CelebA-mapping.txt
```

### Example

```python
from datasetx import CelebAHQ

root = '~/data/CelebA-HQ'  # path to the dataset
train_set = CelebAHQ(root=root, split='train')
valid_set = CelebAHQ(root=root, split='valid')
test_set = CelebAHQ(root=root, split='test')
all_set = CelebAHQ(root=root, split='all')
print(len(train_set))  # 24183
print(len(valid_set))  # 2993
print(len(test_set))   # 2824
print(len(all_set))    # 30000
print(train_set[0]['image'].shape)  # (3, 1024, 1024)
```
