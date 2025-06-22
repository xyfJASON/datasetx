import os
from PIL import Image
from typing import Optional, Callable

from torchvision.datasets import VisionDataset

from .utils import extract_images


class CelebAHQ(VisionDataset):
    """The CelebA-HQ Dataset.

    Please organize the dataset in the following file structure:

    root
    ├── CelebA-HQ-img
    │   ├── 000004.jpg
    │   ├── ...
    │   └── 202591.jpg
    └── CelebA-HQ-to-CelebA-mapping.txt

    The train/valid/test sets are split according to the original CelebA dataset, resulting in 24,183 training images,
    2,993 validation images, and 2,824 test images.

    References:
      - https://github.com/tkarras/progressive_growing_of_gans
      - https://paperswithcode.com/dataset/celeba-hq
      - https://github.com/switchablenorms/CelebAMask-HQ

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root=root, transforms=transforms)

        if split not in ['train', 'valid', 'test', 'all']:
            raise ValueError(f'Invalid split: {split}')
        self.split = split

        # Extract image paths
        image_root = os.path.join(self.root, 'CelebA-HQ-img')
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')

        def filter_func(p):
            if split == 'all':
                return True
            celeba_splits = [1, 162771, 182638, 202600]
            k = 0 if split == 'train' else (1 if split == 'valid' else 2)
            return celeba_splits[k] <= int(os.path.splitext(os.path.basename(p))[0]) < celeba_splits[k+1]

        self.img_paths = extract_images(image_root)
        self.img_paths = list(filter(filter_func, self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        x = Image.open(self.img_paths[index]).convert('RGB')
        if self.transforms is not None:
            x = self.transforms(x)
        return x
