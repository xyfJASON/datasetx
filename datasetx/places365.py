import os
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class Places365(Dataset):
    """The Places365 dataset.

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
            transform_fn: Optional[Callable] = None,
    ):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Invalid split: {split}')
        self.root = os.path.expanduser(root)
        self.split = split
        self.small = small
        self.is_challenge = is_challenge
        self.transform_fn = transform_fn

        # get image root
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

        # load classes
        self.class_to_idx = dict()
        with open(os.path.join(self.root, 'categories_places365.txt'), 'r') as f:
            for line in f:
                name, idx = line.strip().split(' ')
                self.class_to_idx[name] = int(idx)
        self.classes = sorted(self.class_to_idx.keys())

        # load meta file
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

        # extract image paths & labels
        self.image_paths, self.labels = [], []
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
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # read image and label
        x = Image.open(self.image_paths[index]).convert('RGB')
        x = to_tensor(x)
        y = self.labels[index]
        sample = {'image': x, 'label': y}
        # apply transform
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample
