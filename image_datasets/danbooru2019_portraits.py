import os
from PIL import Image
from typing import Optional, Callable

import torchvision.transforms as T
from torch.utils.data import Dataset

from .utils import extract_images


class Danbooru2019Portraits(Dataset):
    """The Danbooru2019 Portraits Dataset.

    The original images have black edges, which can be removed by the provided python script at
    scripts/danbooru_remove_black_edges.py. Note that some images (with too wide black edges) will
    be discarded after processing.

    To load data with this class, the dataset should be organized in the following structure:

    root
    └── portraits
        ├── 10000310.jpg
        └── ...

    This class has one pre-defined transform:
      - 'resize' (default): Resize the image directly to the target size

    References:
      - https://gwern.net/crop#danbooru2019-portraits

    """

    def __init__(
            self,
            root: str,
            img_size: int,
            transform_type: Optional[str] = 'resize',
            transform: Optional[Callable] = None,
    ):
        if transform_type not in ['resize', 'none'] and transform_type is not None:
            raise ValueError(f'Invalid transform_type: {transform_type}')

        root = os.path.expanduser(root)
        image_root = os.path.join(root, 'portraits')
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')

        self.img_size = img_size
        self.transform_type = transform_type
        self.transform = transform
        if transform is None:
            self.transform = self.get_transform()

        self.img_paths = extract_images(image_root)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item])
        if self.transform is not None:
            X = self.transform(X)
        return X

    def get_transform(self):
        if self.transform_type == 'resize':
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'none' or self.transform_type is None:
            transform = None
        else:
            raise ValueError(f'Invalid transform_type: {self.transform_type}')
        return transform
