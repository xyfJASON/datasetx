# CelebA

[Official website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | [Papers with Code](https://paperswithcode.com/dataset/celeba) | [Google drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) | [Baidu drive](https://pan.baidu.com/share/init?surl=CRxxhoQ97A5qbsKO7iaAJg) (password: rp0s)

## Brief introduction

> Copied from official website.

CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including

- 10,177 number of identities,
- 202,599 number of face images, and
- 5 landmark locations, 40 binary attributes annotations per image.

The dataset can be employed as the training and test sets for the following computer vision tasks: face attribute recognition, face recognition, face detection, landmark (or facial part) localization, and face editing & synthesis.

## Statistics

**Numbers**: 202,599

**Splits** (train / valid / test): 162,770 / 19,867 / 19,962

**Resolution**:

- Aligned: 178×218
- In-the-wild: varies from 200+ to 2000+

**Annotations**:

- 10,177 number of identities
- 5 landmark locations per image
- 40 binary attributes annotations

## Files

- Anno
  - identity_CelebA.txt
  - list_attr_celeba.txt
  - list_bbox_celeba.txt
  - list_landmarks_align_celeba.txt
  - list_landmarks_celeba.txt
- Eval
  - list_eval_partition.txt
- Img
  - img_celeba.7z            (In-The-Wild Images)
  - img_align_celeba_png.7z  (Align&Cropped Images, PNG Format)
  - img_align_celeba.zip     (Align&Cropped Images, JPG Format)
- README.txt

## Usage

> The authors provide two versions of dataset: aligned and in-the-wild. `torchvision` only supports loading the aligned version.

### Download

`torchvision` will automatically download the dataset with `download=True`.
Or you can also download the dataset files with `gdown`:

```shell
gdown <file id> -O <filename>
```

| filename                        | file id                           | md5sum                           |
|---------------------------------|-----------------------------------|----------------------------------|
| img_align_celeba.zip            | 0B7EVK8r0v71pZjFTYXZWM3FlRnM      | 00d2c5bc6d35e252742224ab0c1e8fcb |
| identity_CelebA.txt             | 1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS | 32bd1bd63d3c78cd57e08160ec5ed1e2 |
| list_attr_celeba.txt            | 0B7EVK8r0v71pblRyaVFSWGxPY0U      | 75e246fa4810816ffd6ee81facbd244c |
| list_bbox_celeba.txt            | 0B7EVK8r0v71pbThiMVRxWXZ4dU0      | 00566efa6fedff7a56946cd1c10f1c16 |
| list_landmarks_align_celeba.txt | 0B7EVK8r0v71pd0FJY3Blby1HUTQ      | cc24ecafdb5b50baae59b03474781f8c |
| list_landmarks_celeba.txt       | 0B7EVK8r0v71pTzJIdlJWdHczRlU      | 063ee6ddb681f96bc9ca28c6febb9d1a |
| list_eval_partition.txt         | 0B7EVK8r0v71pY0NSMzRuSXJEVkk      | d32c9cbf5e040fd4025c592c306e6668 |

### File structure

```text
root
└── celeba
    ├── identity_CelebA.txt
    ├── list_attr_celeba.txt
    ├── list_bbox_celeba.txt
    ├── list_eval_partition.txt
    ├── list_landmarks_align_celeba.txt
    ├── list_landmarks_celeba.txt
    └── img_align_celeba
        ├── 000001.jpg
        ├── ...
        └── 202599.jpg
```

### Example

```python
from torchvision.datasets import CelebA

train_set = CelebA(root='~/data/CelebA', split='train')
valid_set = CelebA(root='~/data/CelebA', split='valid')
test_set = CelebA(root='~/data/CelebA', split='test')
all_set = CelebA(root='~/data/CelebA', split='all')
print(len(train_set))  # 162770
print(len(valid_set))  # 19867
print(len(test_set))   # 19962
print(len(all_set))    # 202599
```
