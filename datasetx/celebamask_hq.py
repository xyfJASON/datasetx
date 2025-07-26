import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .utils import extract_images


class CelebAMaskHQ(Dataset):
    """The CelebAMask-HQ Dataset.

    Please organize the dataset in the following file structure:

    root
    ├── CelebA-HQ-img
    │   ├── 0.jpg
    │   ├── ...
    │   └── 29999.jpg
    ├── CelebA-HQ-to-CelebA-mapping.txt
    ├── CelebAMask-HQ-attribute-anno.txt
    ├── CelebAMask-HQ-mask-anno
    ├── CelebAMask-HQ-mask
    │   ├── 0.png
    │   ├── ...
    │   └── 29999.png
    ├── CelebAMask-HQ-mask-color
    │   ├── 0.png
    │   ├── ...
    │   └── 29999.png
    ├── CelebAMask-HQ-pose-anno.txt
    └── README.txt

    The train/valid/test sets are split according to the original CelebA dataset, resulting in
    24,183 training images, 2,993 validation images, and 2,824 test images.

    References:
      - https://paperswithcode.com/dataset/celebamask-hq
      - https://github.com/switchablenorms/CelebAMask-HQ

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform_fn: Optional[Callable] = None,
    ):
        if split not in ['train', 'valid', 'test', 'all']:
            raise ValueError(f'Invalid split: {split}')
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform_fn = transform_fn

        # check file structure
        image_root = os.path.join(self.root, 'CelebA-HQ-img')
        mask_root = os.path.join(self.root, 'CelebAMask-HQ-mask')
        mask_color_root = os.path.join(self.root, 'CelebAMask-HQ-mask-color')
        mapping_file = os.path.join(self.root, 'CelebA-HQ-to-CelebA-mapping.txt')
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')
        if not os.path.isdir(mask_root):
            raise ValueError(f'{mask_root} is not an existing directory')
        if not os.path.isdir(mask_color_root):
            raise ValueError(f'{mask_color_root} is not an existing directory')
        if not os.path.isfile(mapping_file):
            raise ValueError(f'{mapping_file} is not an existing file')

        # read the mapping file
        mapping = pd.read_table(mapping_file, sep=r'\s+', index_col=0)
        mapping = {i: int(mapping.iloc[i]['orig_idx']) for i in range(30000)}

        def filter_func(p):
            if split == 'all':
                return True
            orig_idx = mapping[int(os.path.splitext(os.path.basename(p))[0])]
            celeba_splits = [0, 162770, 182637, 202599]
            k = 0 if split == 'train' else (1 if split == 'valid' else 2)
            return celeba_splits[k] <= orig_idx < celeba_splits[k+1]

        # extract image paths
        self.image_paths = extract_images(image_root)
        self.mask_paths = extract_images(mask_root)
        self.mask_color_paths = extract_images(mask_color_root)
        self.image_paths = list(filter(filter_func, self.image_paths))
        self.mask_paths = list(filter(filter_func, self.mask_paths))
        self.mask_color_paths = list(filter(filter_func, self.mask_color_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # read image, mask and mask_color
        x = Image.open(self.image_paths[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index]).convert('L')
        mask_color = Image.open(self.mask_color_paths[index]).convert('RGB')
        # convert to tensor
        x = to_tensor(x)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        mask_color = to_tensor(mask_color)
        sample = {'image': x, 'mask': mask, 'mask_color': mask_color}
        # apply transform
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample
