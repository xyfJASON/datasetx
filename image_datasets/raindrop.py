import os
import random
from PIL import Image
from typing import Optional, Callable

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import VisionDataset

from .utils import extract_images


class Raindrop(VisionDataset):
    """The Raindrop Dataset.

    Raindrop is a set of image pairs, where each pair contains exactly the same background scene, yet one is degraded
    by raindrops and the other one is free from raindrops. To obtain this, the images are captured through two pieces
    of exactly the same glass: one sprayed with water, and the other is left clean. The dataset consists of 1,119 pairs
    of images, with various background scenes and raindrops. They were captured with a Sony A6000 and a Canon EOS 60.
    (Copied from PaperWithCode)

    Please organize the dataset in the following file structure:

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

    References:
      - https://github.com/rui1996/DeRaindrop
      - https://paperswithcode.com/dataset/raindrop

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root=root, transforms=transforms)

        if split not in ['train', 'test', 'test_a', 'test_b']:
            raise ValueError(f'Invalid split: {split}')
        if split == 'test':
            split = 'test_b'
        self.split = split

        # Set the root directory for images and ground truths.
        self.data_root = os.path.join(self.root, self.split, 'data')
        self.gt_root = os.path.join(self.root, self.split, 'gt')
        if not os.path.isdir(self.data_root):
            raise ValueError(f'{self.data_root} is not an existing directory')
        if not os.path.isdir(self.gt_root):
            raise ValueError(f'{self.gt_root} is not an existing directory')

        # Extract image paths
        self.data_paths = extract_images(self.data_root)
        self.gt_paths = extract_images(self.gt_root)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int):
        x = Image.open(self.data_paths[index]).convert('RGB')
        gt = Image.open(self.gt_paths[index]).convert('RGB')
        if self.transforms is not None:
            x, gt = self.transforms(x, gt)
        return x, gt


# ===============================================================================================
# Below are custom transforms that apply to degraded image and ground-truth simultaneously
# Adapted from https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
# ===============================================================================================

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = TF.pad(img, [0, 0, padw, padh], fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = TF.resize(image, self.size, antialias=True)
        target = TF.resize(target, self.size, antialias=True)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            target = TF.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = TF.crop(image, *crop_params)
        target = TF.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = TF.center_crop(image, self.size)
        target = TF.center_crop(target, self.size)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        target = TF.to_tensor(target)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        target = TF.normalize(target, mean=self.mean, std=self.std)
        return image, target
