import io
import os
import tarfile
import numpy as np
from PIL import Image
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.datasets import VisionDataset


class MarigoldDepthEval(VisionDataset):
    """Dataset for evaluating Affine-Invariant Depth Estimation.

    Please organize the dataset in the following file structure:

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

    Reference:
      - https://github.com/prs-eth/Marigold/tree/main/src/dataset
      - https://github.com/EnVision-Research/Lotus/tree/main/evaluation/dataset_depth

    """

    def __init__(
            self,
            root: str,
            dataset: str = 'nyuv2',
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root=root, transforms=transforms)
        self.dataset = dataset

        if self.dataset.lower() == 'nyuv2':
            tar_file = os.path.join(self.root, 'nyuv2', 'nyu_labeled_extracted.tar')
            txt_file = os.path.join(self.root, 'nyuv2', 'filename_list_test.txt')
            self.min_depth = 1e-3
            self.max_depth = 10.0
        elif self.dataset.lower() == 'kitti':
            tar_file = os.path.join(self.root, 'kitti', 'kitti_eigen_split_test.tar')
            txt_file = os.path.join(self.root, 'kitti', 'eigen_test_files_with_gt.txt')
            self.min_depth = 1e-5
            self.max_depth = 80
        elif self.dataset.lower() == 'eth3d':
            tar_file = os.path.join(self.root, 'eth3d', 'eth3d.tar')
            txt_file = os.path.join(self.root, 'eth3d', 'eth3d_filename_list.txt')
            self.min_depth = 1e-5
            self.max_depth = torch.inf
        elif self.dataset.lower() == 'scannet':
            tar_file = os.path.join(self.root, 'scannet', 'scannet_val_sampled_800_1.tar')
            txt_file = os.path.join(self.root, 'scannet', 'scannet_val_sampled_list_800_1.txt')
            self.min_depth = 1e-3
            self.max_depth = 10.0
        elif self.dataset.lower() == 'diode':
            tar_file = os.path.join(self.root, 'diode', 'diode_val.tar')
            txt_file = os.path.join(self.root, 'diode', 'diode_val_all_filename_list.txt')
            self.min_depth = 0.6
            self.max_depth = 350
        else:
            raise ValueError(f'Unsupported dataset: {self.dataset}')

        with open(txt_file, 'r') as f:
            self.filenames = [s.split() for s in f.readlines()]
        if self.dataset.lower() == 'kitti':
            self.filenames = [f for f in self.filenames if f[1] != 'None']

        self.tar_obj = tarfile.open(tar_file)

    def __len__(self):
        return len(self.filenames)

    def __del__(self):
        self.tar_obj.close()

    def __getitem__(self, index: int):
        rgb_rel_path = self.filenames[index][0]
        depth_rel_path = self.filenames[index][1]

        # read image, (3, H, W), torch.float32, [0, 1]
        image = self.tar_obj.extractfile('./' + rgb_rel_path).read()
        image = np.array(Image.open(io.BytesIO(image)))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.

        # read depth, (1, H, W), torch.float32, [0, inf]
        depth = self.tar_obj.extractfile('./' + depth_rel_path).read()
        if self.dataset.lower() == 'nyuv2':
            depth = np.array(Image.open(io.BytesIO(depth))) / 1000.0
        elif self.dataset.lower() == 'kitti':
            depth = np.array(Image.open(io.BytesIO(depth))) / 256.0
        elif self.dataset.lower() == 'eth3d':
            depth = np.frombuffer(depth, dtype=np.float32).copy()
            depth[depth == torch.inf] = 0.0
            depth = depth.reshape((4032, 6048))
        elif self.dataset.lower() == 'scannet':
            depth = np.array(Image.open(io.BytesIO(depth))) / 1000.0
        elif self.dataset.lower() == 'diode':
            depth = np.load(io.BytesIO(depth))
        depth = torch.from_numpy(depth.squeeze()[None, :, :]).float()

        # kitti: crop to benchmark size
        if self.dataset.lower() == 'kitti':
            KB_CROP_HEIGHT = 352
            KB_CROP_WIDTH = 1216
            height, width = depth.shape[-2:]
            top_margin = int(height - KB_CROP_HEIGHT)
            left_margin = int((width - KB_CROP_WIDTH) / 2)
            image = image[:, top_margin:top_margin + KB_CROP_HEIGHT, left_margin:left_margin + KB_CROP_WIDTH]
            depth = depth[:, top_margin:top_margin + KB_CROP_HEIGHT, left_margin:left_margin + KB_CROP_WIDTH]

        # get mask, (1, H, W), torch.bool
        mask = torch.logical_and((depth > self.min_depth), (depth < self.max_depth)).bool()
        if self.dataset.lower() == 'nyuv2':
            eval_mask = torch.zeros_like(mask).bool()
            eval_mask[:, 45:471, 41:601] = 1
            mask = torch.logical_and(mask, eval_mask)
        elif self.dataset.lower() == 'kitti':
            eval_mask = torch.zeros_like(mask).bool()
            _, gt_height, gt_width = eval_mask.shape
            eval_mask[:,
                int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
            ] = 1
            mask = torch.logical_and(mask, eval_mask)
        elif self.dataset.lower() == 'diode':
            mask_rel_path = self.filenames[index][2]
            mask = self.tar_obj.extractfile('./' + mask_rel_path).read()
            mask = np.load(io.BytesIO(mask)).squeeze()[None, :, :].astype(bool)
            mask = torch.from_numpy(mask).bool()

        # apply transforms
        if self.transforms is not None:
            image, depth, mask = self.transforms(image, depth, mask)

        return image, depth, mask


# ===============================================================================================
# Below are custom transforms that apply to image, depth and mask simultaneously
# Adapted from https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
# ===============================================================================================

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, depth, mask):
        for t in self.transforms:
            image, depth, mask = t(image, depth, mask)
        return image, depth, mask


class Resize:
    def __init__(self, size):
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

    def __call__(self, image, depth, mask):
        image = F.interpolate(image.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False, antialias=True).squeeze(0)
        depth = F.interpolate(depth.unsqueeze(0), size=self.size, mode='nearest').squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=self.size, mode='nearest').squeeze(0) > 0.5
        return image, depth, mask


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, depth, mask):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, depth, mask
