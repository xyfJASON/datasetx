import os
from PIL import Image
from typing import Optional, Callable

import torchvision.transforms as T
from torch.utils.data import Dataset

from .utils import extract_images


class AFHQ(Dataset):
    """The Animal Faces-HQ (AFHQ) Dataset.

    Animal FacesHQ (AFHQ) is a dataset of animal faces consisting of 15,000 high-quality images at 512 × 512
    resolution. The dataset includes three domains of cat, dog, and wildlife, each providing 5000 images.
    By having multiple (three) domains and diverse images of various breeds (≥ eight) per each domain, AFHQ
    sets a more challenging image-to-image translation problem. All images are vertically and horizontally
    aligned to have the eyes at the center. The low-quality images were discarded by human effort.
    (Copied from PapersWithCode)

    To load data with this class, the dataset should be organized in the following structure:

    root
    ├── train
    │   ├── cat        (contains 5065 images)
    │   ├── dog        (contains 4678 images)
    │   └── wild       (contains 4593 images)
    └── test
        ├── cat        (contains 493 images)
        ├── dog        (contains 491 images)
        └── wild       (contains 483 images)

    This class has one pre-defined transform:
      - 'resize' (default): Resize the image directly to the target size

    References:
      - https://github.com/clovaai/stargan-v2
      - https://paperswithcode.com/dataset/afhq
      - https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0

    """

    def __init__(
            self,
            root: str,
            img_size: int,
            split: str = 'train',
            transform_type: str = 'default',
            transform: Optional[Callable] = None,
    ):
        if split not in ['train', 'test']:
            raise ValueError(f'Invalid split: {split}')
        root = os.path.expanduser(root)
        image_root = os.path.join(root, split)
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')

        self.img_size = img_size
        self.split = split
        self.transform_type = transform_type
        self.transform = transform
        if transform is None:
            self.transform = self.get_transform()

        self.img_paths = extract_images(image_root)
        self.labels = []
        for p in self.img_paths:
            if 'cat' in p:
                self.labels.append(0)
            elif 'dog' in p:
                self.labels.append(1)
            elif 'wild' in p:
                self.labels.append(2)
            else:
                raise ValueError(f'Invalid label: {p}')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item])
        if self.transform is not None:
            X = self.transform(X)
        y = self.labels[item]
        return X, y

    def get_transform(self):
        flip_p = 0.5 if self.split == 'train' else 0.0
        if self.transform_type in ['default', 'resize']:
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'none':
            transform = None
        else:
            raise ValueError(f'Invalid transform_type: {self.transform_type}')
        return transform
