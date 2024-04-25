import os
from PIL import Image
from typing import Optional, Callable

from torchvision.datasets import VisionDataset

from .utils import extract_images


class Danbooru2019Portraits(VisionDataset):
    """The Danbooru2019 Portraits Dataset.

    The original images have black edges, which can be removed by the provided python script at
    `scripts/danbooru_remove_black_edges.py`. Note that some images (with too wide black edges) will
    be discarded after processing.

    Please organize the dataset in the following file structure:

    root
    └── portraits
        ├── 10000310.jpg
        └── ...

    References:
      - https://gwern.net/crop#danbooru2019-portraits
      - https://github.com/LynnHo/EigenGAN-Tensorflow/blob/main/scripts/remove_black_edge.py

    """

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root=root, transforms=transforms)

        # Extract image paths
        image_root = os.path.join(self.root, 'portraits')
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
