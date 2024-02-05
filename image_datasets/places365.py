import os
from PIL import Image
from typing import Optional, Callable

import torchvision.transforms as T
from torch.utils.data import Dataset


class Places365(Dataset):
    """The Places365 dataset.

    The Places365 dataset is a scene recognition dataset. It is composed of 10 million images comprising 434 scene
    classes. There are two versions of the dataset: Places365-Standard with 1.8 million train and 36000 validation
    images from K=365 scene classes, and Places365-Challenge-2016, in which the size of the training set is increased
    up to 6.2 million extra images, including 69 new scene classes (leading to a total of 8 million train images from
    434 scene classes). (Copied from PapersWithCode)

    This class has two pre-defined transforms:
      - 'resize-crop' (default): Resize the image so that the short side match the target size, then crop a square patch
      - 'resize': Resize the image directly to the target size
    All of the above transforms will be followed by random horizontal flipping.

    To load data with this class, the dataset should be organized in the following structure:

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
            img_size: int,
            split: str = 'train',
            small: bool = False,
            is_challenge: bool = False,
            transform_type: Optional[str] = 'resize-crop',
            transform: Optional[Callable] = None,
    ):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Invalid split: {split}')
        if transform_type not in ['resize-crop', 'resize', 'none'] and transform_type is not None:
            raise ValueError(f'Invalid transform_type: {transform_type}')

        size = '256' if small else 'large'
        variant = 'challenge' if is_challenge else 'standard'
        root = os.path.expanduser(root)
        if split == 'train':
            img_root = os.path.join(root, f'data_{size}_{variant}')
        elif split == 'valid':
            img_root = os.path.join(root, f'val_{size}')
        elif split == 'test':
            img_root = os.path.join(root, f'test_{size}')
        else:
            raise ValueError
        assert os.path.isdir(img_root), f'{img_root} is not an existing directory'

        self.root = root
        self.img_size = img_size
        self.split = split
        self.transform_type = transform_type
        self.transform = transform
        if transform is None:
            self.transform = self.get_transform()

        # classes
        self.class_to_idx = dict()
        with open(os.path.join(root, 'categories_places365.txt'), 'r') as f:
            for line in f:
                name, idx = line.strip().split(' ')
                self.class_to_idx[name] = int(idx)
        self.classes = sorted(self.class_to_idx.keys())

        # meta file
        if split == 'train':
            metafile = os.path.join(root, f'places365_train_{variant}.txt')
        elif split == 'valid':
            metafile = os.path.join(root, 'places365_val.txt')
        elif split == 'test':
            metafile = os.path.join(root, 'places365_test.txt')
        else:
            raise ValueError
        assert os.path.isfile(metafile), f'{metafile} is not an existing file'

        # images & labels
        self.img_paths, self.labels = [], []
        with open(metafile, 'r') as f:
            for line in f:
                if split in ['train', 'valid']:
                    img_path, label = line.strip().split(' ')
                    label = int(label)
                else:
                    img_path, label = line.strip(), None
                img_path = os.path.join(img_root, img_path.lstrip('/'))
                assert os.path.isfile(img_path), f'{img_path} is not an existing file'
                self.img_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item]).convert('RGB')
        if self.transform is not None:
            X = self.transform(X)
        return X, self.labels[item]

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
