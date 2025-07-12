# Marigold-Depth-Eval

[Official website](https://github.com/prs-eth/Marigold) | [Official drive](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/)

## Brief introduction

The datasets used in [Marigold](https://arxiv.org/abs/2312.02145) and some other papers for evaluating affine-invariant depth estimation.
It includes 5 sub-datasets: NYUv2, KITTI, ETH3D, ScanNet, and DIODE.

## Statistics

| Dataset | Numbers | Resolution |
|:-------:|:-------:|:----------:|
|  NYUv2  |   654   |  640×480   |
|  KITTI  |   652   | 1216×352\* |
|  ETH3D  |   454   | 6048×4032  |
| ScanNet |   800   |  640×480   |
|  DIODE  |   771   |  1024×768  |

\* Note: KITTI will be cropped to 1216×352 for benchmarking.

## Usage

### Download

```shell
wget -r -np -nH --cut-dirs=4 -R "index.html*" https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/

PREFIX="https://raw.githubusercontent.com/EnVision-Research/Lotus/refs/heads/main/datasets/eval/depth/data_split"
wget ${PREFIX}/nyu/labeled/filename_list_test.txt -P nyuv2
wget ${PREFIX}/nyu/labeled/filename_list_train.txt -P nyuv2
wget ${PREFIX}/kitti/eigen_test_files_with_gt.txt -P kitti
wget ${PREFIX}/kitti/eigen_val_from_train_800.txt -P kitti
wget ${PREFIX}/eth3d/eth3d_filename_list.txt -P eth3d
wget ${PREFIX}/scannet/scannet_val_sampled_list_800_1.txt -P scannet
wget ${PREFIX}/diode/diode_val_all_filename_list.txt -P diode
wget ${PREFIX}/diode/diode_val_indoor_filename_list.txt -P diode
wget ${PREFIX}/diode/diode_val_outdoor_filename_list.txt -P diode
```

### File structure

> The dataset can read images directly from the tar files, so you don't need to extract them.

```text
root
├── diode
│   ├── diode_val_all_filename_list.txt
│   ├── diode_val_indoor_filename_list.txt
│   ├── diode_val_outdoor_filename_list.txt
│   └── diode_val.tar
├── eth3d
│   ├── eth3d_filename_list.txt
│   └── eth3d.tar
├── kitti
│   ├── eigen_test_files_with_gt.txt
│   ├── eigen_val_from_train_800.txt
│   ├── kitti_eigen_split_test.tar
│   └── kitti_sampled_val_800.tar
├── nyuv2
│   ├── filename_list_test.txt
│   ├── filename_list_train.txt
│   └── nyu_labeled_extracted.tar
└── scannet
    ├── scannet_val_sampled_list_800_1.txt
    └── scannet_val_sampled_800_1.tar
```

### Example

```python
from image_datasets import MarigoldDepthEval

root = '~/data/marigold'  # path to the dataset
dsine_eval_nyuv2 = MarigoldDepthEval(root=root, dataset='nyuv2')
dsine_eval_kitti = MarigoldDepthEval(root=root, dataset='kitti')
dsine_eval_eth3d = MarigoldDepthEval(root=root, dataset='eth3d')
dsine_eval_scannet = MarigoldDepthEval(root=root, dataset='scannet')
dsine_eval_diode = MarigoldDepthEval(root=root, dataset='diode')

print(len(dsine_eval_nyuv2))    # 654
print(len(dsine_eval_kitti))    # 697
print(len(dsine_eval_eth3d))    # 454
print(len(dsine_eval_scannet))  # 800
print(len(dsine_eval_diode))    # 771

image, depth, mask = dsine_eval_nyuv2[0]
print(image.shape, depth.shape, mask.shape)  # (3, 480, 640) (1, 480, 640) (1, 480, 640)
image, depth, mask = dsine_eval_kitti[0]
print(image.shape, depth.shape, mask.shape)  # (3, 352, 1216) (1, 352, 1216) (1, 352, 1216)
image, depth, mask = dsine_eval_eth3d[0]
print(image.shape, depth.shape, mask.shape)  # (3, 4032, 6048) (1, 4032, 6048) (1, 4032, 6048)
image, depth, mask = dsine_eval_scannet[0]
print(image.shape, depth.shape, mask.shape)  # (3, 480, 640) (1, 480, 640) (1, 480, 640)
image, depth, mask = dsine_eval_diode[0]
print(image.shape, depth.shape, mask.shape)  # (3, 768, 1024) (1, 768, 1024) (1, 768, 1024)
```
