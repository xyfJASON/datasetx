import os
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .utils import extract_images


class Raindrop(Dataset):
    """The Raindrop Dataset.

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
            transform_fn: Optional[Callable] = None,
    ):
        if split not in ['train', 'test', 'test_a', 'test_b']:
            raise ValueError(f'Invalid split: {split}')
        self.root = os.path.expanduser(root)
        self.split = 'test_b' if split == 'test' else split
        self.transform_fn = transform_fn

        # set the root directory for images and ground truths
        self.data_root = os.path.join(self.root, self.split, 'data')
        self.gt_root = os.path.join(self.root, self.split, 'gt')
        if not os.path.isdir(self.data_root):
            raise ValueError(f'{self.data_root} is not an existing directory')
        if not os.path.isdir(self.gt_root):
            raise ValueError(f'{self.gt_root} is not an existing directory')

        # extract image paths
        self.data_paths = extract_images(self.data_root)
        self.gt_paths = extract_images(self.gt_root)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int):
        # read image and ground truth
        x = Image.open(self.data_paths[index]).convert('RGB')
        gt = Image.open(self.gt_paths[index]).convert('RGB')
        # convert to tensor
        x = to_tensor(x)
        gt = to_tensor(gt)
        sample = {'image': x, 'gt': gt}
        # apply transform
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample
