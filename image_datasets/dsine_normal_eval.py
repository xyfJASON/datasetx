import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2
import pickle
import numpy as np
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.datasets import VisionDataset


class DSINENormalEval(VisionDataset):
    """Dataset for evaluating Surface Normal Estimation.

    Please organize the dataset in the following file structure:

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

    References:
      - https://github.com/baegwangbin/DSINE
      - https://github.com/EnVision-Research/Lotus/tree/main/evaluation/dataset_normal

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
            self.image_paths = list(map(str, Path(os.path.join(self.root, 'nyuv2', 'test')).rglob('*_img.png')))
        elif self.dataset.lower() == 'scannet':
            self.image_paths = list(map(str, Path(os.path.join(self.root, 'scannet')).rglob('*_img.png')))
        elif self.dataset.lower() == 'ibims':
            self.image_paths = list(map(str, Path(os.path.join(self.root, 'ibims', 'ibims')).rglob('*_img.png')))
        elif self.dataset.lower() == 'sintel':
            self.image_paths = list(map(str, Path(os.path.join(self.root, 'sintel')).rglob('*_img.png')))
        elif self.dataset.lower() == 'vkitti':
            self.image_paths = list(map(str, Path(os.path.join(self.root, 'vkitti')).rglob('*_img.jpg')))
        elif self.dataset.lower() == 'oasis':
            self.image_paths = list(map(str, Path(os.path.join(self.root, 'oasis', 'val')).rglob('*_img.png')))
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        # read image (H, W, 3), np.float32, [0, 1]
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # read normal (H, W, 3), np.float32, [-1, 1]
        # and mask (H, W, 1), np.bool
        if self.dataset.lower() in ['nyuv2', 'scannet']:
            normal_path = image_path.replace('_img.png', '_normal.png')
            normal = cv2.cvtColor(cv2.imread(normal_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            mask = np.sum(normal, axis=2, keepdims=True) > 0
            normal = (normal.astype(np.float32) / 255.0) * 2.0 - 1.0
        elif self.dataset.lower() in ['ibims', 'sintel']:
            normal_path = image_path.replace('_img.png', '_normal.exr')
            normal = cv2.cvtColor(cv2.imread(normal_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            mask = np.linalg.norm(normal, axis=2, keepdims=True) > 0.5
        elif self.dataset.lower() == 'vkitti':
            normal_path = image_path.replace('_img.jpg', '_normal.png')
            normal = cv2.cvtColor(cv2.imread(normal_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            mask = np.sum(normal, axis=2, keepdims=True) > 0
            normal = (normal.astype(np.float32) / 65535.0) * 2.0 - 1.0
        elif self.dataset.lower() == 'oasis':
            normal_path = image_path.replace('_img.png', '_normal.pkl')
            h, w = image.shape[:2]
            normal_dict = pickle.load(open(normal_path, 'rb'))
            normal, mask = np.zeros((h, w, 3)), np.zeros((h, w))
            # stuff ROI normal into bounding box
            min_y = normal_dict['min_y']
            max_y = normal_dict['max_y']
            min_x = normal_dict['min_x']
            max_x = normal_dict['max_x']
            roi_normal = normal_dict['normal']
            # to LUB
            normal[min_y:max_y+1, min_x:max_x+1, :] = roi_normal
            normal = normal.astype(np.float32)
            normal[:,:,0] *= -1
            normal[:,:,1] *= -1
            # make mask
            roi_mask = np.logical_or(
                np.logical_or(roi_normal[:,:,0] != 0, roi_normal[:,:,1] != 0), roi_normal[:,:,2] != 0
            ).astype(np.float32)
            mask[min_y:max_y+1, min_x:max_x+1] = roi_mask
            mask = mask[:, :, None]
            mask = mask > 0.5
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        # to tensor: image (3, H, W), normal (3, H, W), mask (1, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        normal = torch.from_numpy(normal).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).bool()

        # apply transforms
        if self.transforms is not None:
            image, normal, mask = self.transforms(image, normal, mask)

        return image, normal, mask


# ===============================================================================================
# Below are custom transforms that apply to image, normal and mask simultaneously
# Adapted from https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
# ===============================================================================================

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, normal, mask):
        for t in self.transforms:
            image, normal, mask = t(image, normal, mask)
        return image, normal, mask


class Resize:
    def __init__(self, size):
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

    def __call__(self, image, normal, mask):
        image = F.interpolate(image.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False, antialias=True).squeeze(0)
        normal = F.interpolate(normal.unsqueeze(0), size=self.size, mode='nearest').squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=self.size, mode='nearest').squeeze(0) > 0.5
        return image, normal, mask


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, normal, mask):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, normal, mask
