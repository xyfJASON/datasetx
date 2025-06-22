import os
from PIL import Image
from typing import Optional, Callable

from torchvision.datasets import VisionDataset

from .utils import extract_images


class AFHQ(VisionDataset):
    """The Animal Faces-HQ (AFHQ) Dataset.

    Please organize the dataset in the following file structure:

    root
    ├── train
    │   ├── cat        (contains 5065 images)
    │   ├── dog        (contains 4678 images)
    │   └── wild       (contains 4593 images)
    └── test
        ├── cat        (contains 493 images)
        ├── dog        (contains 491 images)
        └── wild       (contains 483 images)

    References:
      - https://github.com/clovaai/stargan-v2
      - https://paperswithcode.com/dataset/afhq
      - https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root=root, transforms=transforms)

        if split not in ['train', 'test']:
            raise ValueError(f'Invalid split: {split}')
        self.split = split

        # Extract image paths
        image_root = os.path.join(self.root, split)
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')
        self.img_paths = extract_images(image_root)

        # Extract labels
        self.labels = []
        for p in self.img_paths:
            self.labels.append(0 if 'cat' in p else 1 if 'dog' in p else 2)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        x = Image.open(self.img_paths[index]).convert('RGB')
        if self.transforms is not None:
            x = self.transforms(x)
        y = self.labels[index]
        return x, y
