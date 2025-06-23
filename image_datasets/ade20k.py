import os
import random
import numpy as np
from PIL import Image
from typing import Callable, Optional

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import VisionDataset


class SceneParse150(VisionDataset):
    """The ADE20K SceneParse150 Dataset.

    Please organize the dataset in the following file structure:

    root
    ├── ADEChallengeData2016
    │   ├── annotations
    │   │   ├── training
    │   │   └── validation
    │   ├── images
    │   │   ├── training
    │   │   └── validation
    │   ├── objectInfo150.txt
    │   └── sceneCategories.txt
    └── release_test
        ├── list.txt
        ├── readme.txt
        └── testing

    References:
      - https://ade20k.csail.mit.edu/
      - http://sceneparsing.csail.mit.edu/
      - https://paperswithcode.com/dataset/ade20k

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root=root, transforms=transforms)

        if split in ['train', 'training']:
            self.images_root = os.path.join(self.root, 'ADEChallengeData2016', 'images', 'training')
            self.annotations_root = os.path.join(self.root, 'ADEChallengeData2016', 'annotations', 'training')
        elif split in ['val', 'valid', 'validation']:
            self.images_root = os.path.join(self.root, 'ADEChallengeData2016', 'images', 'validation')
            self.annotations_root = os.path.join(self.root, 'ADEChallengeData2016', 'annotations', 'validation')
        elif split in ['test', 'testing']:
            self.images_root = os.path.join(self.root, 'release_test', 'testing')
            self.annotations_root = None

        self.filenames = [f.replace('.jpg', '') for f in os.listdir(self.images_root) if f.endswith('.jpg')]
        self.filenames = list(sorted(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        image_path = os.path.join(self.images_root, self.filenames[index] + '.jpg')
        image = Image.open(image_path).convert('RGB')
        annotation = None
        if self.annotations_root is not None:
            annotation_path = os.path.join(self.annotations_root, self.filenames[index] + '.png')
            annotation = Image.open(annotation_path).convert('L')
        if self.transforms is not None:
            image, annotation = self.transforms(image, annotation)
        return image, annotation


# ===============================================================================================
# Below are custom transforms that apply to image and mask simultaneously
# Adapted from https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
# ===============================================================================================

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, mask):
        image = TF.resize(image, self.size, antialias=True)
        mask = TF.resize(mask, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, mask):
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class ToTensor:
    def __call__(self, image, mask):
        image = TF.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        return image, mask


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, mask
