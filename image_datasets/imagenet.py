import os
from PIL import Image
from typing import Optional, Callable

from torchvision.datasets import VisionDataset

from .utils import extract_images


class ImageNet(VisionDataset):
    """The ImageNet-1K (ILSVRC 2012) Dataset.

    Please organize the dataset in the following file structure:

    root
    ├── train
    │   ├── n01440764
    │   ├── ...
    │   └── n15075141
    ├── val (or valid)
    │   ├── ILSVRC2012_val_00000001.JPEG
    │   ├── ...
    │   └── ILSVRC2012_val_00050000.JPEG
    └── test
        ├── ILSVRC2012_test_00000001.JPEG
        ├── ...
        └── ILSVRC2012_test_00100000.JPEG

    References:
      - https://image-net.org/challenges/LSVRC/2012/2012-downloads.php

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root=root, transforms=transforms)

        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Invalid split: {split}')
        self.split = split

        # Extract image paths
        image_root = os.path.join(self.root, split)
        if split == 'valid' and not os.path.isdir(image_root):
            image_root = os.path.join(self.root, 'val')
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')
        self.img_paths = extract_images(image_root)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        x = Image.open(self.img_paths[index]).convert('RGB')
        if self.transforms is not None:
            x = self.transforms(x)
        return x
