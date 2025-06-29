# DSINE-Normal-Eval

[Official website](https://github.com/baegwangbin/DSINE) | [Google drive](https://drive.google.com/drive/folders/1t3LMJIIrSnCGwOEf53Cyg0lkSXd3M4Hm?usp=drive_link)

## Brief introduction

The datasets used in [DSINE](https://arxiv.org/abs/2403.00712) and some other papers for evaluating surface normal estimation.
It includes 6 sub-datasets: NYUv2, ScanNet, iBims, Sintel, VKITTI, and OASIS.

## Statistics

| Dataset | Numbers | Resolution |
|:-------:|:-------:|:----------:|
|  NYUv2  |   654   |  640×480   |
| ScanNet |   300   |  640×480   |
|  iBims  |   100   |  640×480   |
| Sintel  |  1,064  |  1024×436  |
| VKITTI  |  1,000  |  1242×375  |
|  OASIS  | 10,000  | ~1024×863  |

## Usage

### Download

```shell
gdown 1u6_-Zvg7ufKwlq0ManQgL-xJL_DwUCR_ -O dsine_eval.zip
unzip dsine_eval.zip
```

### File structure

```text
root
├── ibims
│   ├── ibims
│   └── readme.txt
├── nyuv2
│   ├── readme.txt
│   ├── train
│   └── test
├── oasis
│   ├── readme.txt
│   └── val
├── scannet
│   ├── readme.txt
│   ├── scene0001_01
│   ├── ...
│   └── scene0806_00
├── sintel
│   ├── readme.txt
│   ├── alley_1
│   ├── ...
│   └── temple_3
└── vkitti
    ├── readme.txt
    ├── Scene01
    ├── ...
    └── Scene20
```

### Example

```python
from image_datasets import DSINENormalEval

root = '~/data/dsine_eval'  # path to the dataset
dsine_eval_nyuv2 = DSINENormalEval(root=root, dataset='nyuv2')
dsine_eval_scannet = DSINENormalEval(root=root, dataset='scannet')
dsine_eval_ibims = DSINENormalEval(root=root, dataset='ibims')
dsine_eval_sintel = DSINENormalEval(root=root, dataset='sintel')
dsine_eval_vkitti = DSINENormalEval(root=root, dataset='vkitti')
dsine_eval_oasis = DSINENormalEval(root=root, dataset='oasis')

print(len(dsine_eval_nyuv2))  # 654
print(len(dsine_eval_scannet))  # 300
print(len(dsine_eval_ibims))  # 100
print(len(dsine_eval_sintel))  # 1064
print(len(dsine_eval_vkitti))  # 1000
print(len(dsine_eval_oasis))  # 10000

image, normal, mask = dsine_eval_nyuv2[0]
print(image.shape, normal.shape, mask.shape)  # (3, 480, 640) (3, 480, 640) (1, 480, 640)

image, normal, mask = dsine_eval_scannet[0]
print(image.shape, normal.shape, mask.shape)  # (3, 480, 640) (3, 480, 640) (1, 480, 640)

image, normal, mask = dsine_eval_ibims[0]
print(image.shape, normal.shape, mask.shape)  # (3, 480, 640) (3, 480, 640) (1, 480, 640)

image, normal, mask = dsine_eval_sintel[0]
print(image.shape, normal.shape, mask.shape)  # (3, 436, 1024) (3, 436, 1024) (1, 436, 1024)

image, normal, mask = dsine_eval_vkitti[0]
print(image.shape, normal.shape, mask.shape)  # (3, 375, 1242) (3, 375, 1242) (1, 375, 1242)

image, normal, mask = dsine_eval_oasis[0]
print(image.shape, normal.shape, mask.shape)  # (3, 607, 1024) (3, 607, 1024) (1, 607, 1024)
```
