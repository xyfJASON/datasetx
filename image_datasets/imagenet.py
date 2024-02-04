import os
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
import torchvision.transforms as T

from .utils import extract_images


class ImageNet(Dataset):
    """Extend torchvision.datasets.ImageNet with two pre-defined transforms and support test set.

    This class has two pre-defined transforms:
      - 'resize-crop' (default): Resize the image so that the short side match the target size, then crop a square patch
      - 'resize': Resize the image directly to the target size
    All of the above transforms will be followed by random horizontal flipping.

    To load data with this class, the dataset should be organized in the following structure:

    root
    ├── train
    │   ├── n01440764
    │   ├── ...
    │   └── n15075141
    ├── valid (or val)
    │   ├── n01440764    (or directly put all validation images here)
    │   ├── ...
    │   └── n15075141
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
            img_size: int,
            split: str = 'train',
            transform_type: Optional[str] = 'resize-crop',
            transform: Optional[Callable] = None,
    ):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Invalid split: {split}')
        if transform_type not in ['resize-crop', 'resize', 'none'] and transform_type is not None:
            raise ValueError(f'Invalid transform_type: {transform_type}')

        root = os.path.expanduser(root)
        image_root = os.path.join(root, split)
        if split == 'valid' and not os.path.isdir(image_root):
            image_root = os.path.join(root, 'val')
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')

        self.img_size = img_size
        self.split = split
        self.transform_type = transform_type
        self.transform = transform

        self.img_paths = extract_images(image_root)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item]).convert('RGB')
        if self.transform is not None:
            X = self.transform(X)
        return X

    def get_transform(self):
        crop = T.RandomCrop if self.split == 'train' else T.CenterCrop
        flip_p = 0.5 if self.split == 'train' else 0.0
        if self.transform_type == 'resize-crop':
            transform = T.Compose([
                T.Resize(self.img_size, antialias=True),
                crop((self.img_size, self.img_size)),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'resize':
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'none' or self.transform_type is None:
            transform = None
        else:
            raise ValueError(f'Invalid transform_type: {self.transform_type}')
        return transform
