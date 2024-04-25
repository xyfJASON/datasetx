import os
from PIL import Image
from typing import Optional, Callable

from torchvision.datasets import VisionDataset


class Places365(VisionDataset):
    """The Places365 dataset.

    The Places365 dataset is a scene recognition dataset. It is composed of 10 million images comprising 434 scene
    classes. There are two versions of the dataset: Places365-Standard with 1.8 million train and 36000 validation
    images from K=365 scene classes, and Places365-Challenge-2016, in which the size of the training set is increased
    up to 6.2 million extra images, including 69 new scene classes (leading to a total of 8 million train images from
    434 scene classes). (Copied from PapersWithCode)

    Please organize the dataset in the following file structure:

    root
    ├── categories_places365.txt
    ├── places365_train_standard.txt
    ├── places365_train_challenge.txt
    ├── places365_val.txt
    ├── places365_test.txt
    ├── data_256_standard
    │   ├── a
    │   ├── ...
    │   └── z
    ├── data_large_standard
    │   ├── a
    │   ├── ...
    │   └── z
    ├── val_256
    │   ├── Places365_val_00000001.jpg
    │   ├── ...
    │   └── Places365_val_00036500.jpg
    ├── val_large
    │   ├── Places365_val_00000001.jpg
    │   ├── ...
    │   └── Places365_val_00036500.jpg
    ├── test_256
    │   ├── Places365_test_00000001.jpg
    │   ├── ...
    │   └── Places365_test_00328500.jpg
    └── test_large
        ├── Places365_test_00000001.jpg
        ├── ...
        └── Places365_test_00328500.jpg

    References:
      - http://places2.csail.mit.edu/index.html
      - https://paperswithcode.com/dataset/places365

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            small: bool = False,
            is_challenge: bool = False,
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root=root, transforms=transforms)

        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Invalid split: {split}')
        self.split = split
        self.small = small
        self.is_challenge = is_challenge

        # Get image root
        size = '256' if small else 'large'
        variant = 'challenge' if is_challenge else 'standard'
        if split == 'train':
            img_root = os.path.join(self.root, f'data_{size}_{variant}')
        elif split == 'valid':
            img_root = os.path.join(self.root, f'val_{size}')
        elif split == 'test':
            img_root = os.path.join(self.root, f'test_{size}')
        else:
            raise ValueError(f'Invalid split: {split}')
        if not os.path.isdir(img_root):
            raise ValueError(f'{img_root} is not an existing directory')

        # Load classes
        self.class_to_idx = dict()
        with open(os.path.join(self.root, 'categories_places365.txt'), 'r') as f:
            for line in f:
                name, idx = line.strip().split(' ')
                self.class_to_idx[name] = int(idx)
        self.classes = sorted(self.class_to_idx.keys())

        # Load meta file
        if split == 'train':
            metafile = os.path.join(self.root, f'places365_train_{variant}.txt')
        elif split == 'valid':
            metafile = os.path.join(self.root, 'places365_val.txt')
        elif split == 'test':
            metafile = os.path.join(self.root, 'places365_test.txt')
        else:
            raise ValueError(f'Invalid split: {split}')
        if not os.path.isfile(metafile):
            raise ValueError(f'{metafile} is not an existing file')

        # Extract image paths & labels
        self.img_paths, self.labels = [], []
        with open(metafile, 'r') as f:
            for line in f:
                if split in ['train', 'valid']:
                    img_path, label = line.strip().split(' ')
                    label = int(label)
                else:
                    img_path, label = line.strip(), None
                img_path = os.path.join(img_root, img_path.lstrip('/'))
                if not os.path.isfile(img_path):
                    raise ValueError(f'{img_path} is not an existing file')
                self.img_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        x = Image.open(self.img_paths[index]).convert('RGB')
        if self.transforms is not None:
            x = self.transforms(x)
        return x, self.labels[index]
