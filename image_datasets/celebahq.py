import os
from PIL import Image
from typing import Optional, Callable

from torchvision.datasets import VisionDataset

from .utils import extract_images


class CelebAHQ(VisionDataset):
    """The CelebA-HQ Dataset.

    The CelebA-HQ dataset is a high-quality version of CelebA that consists of 30,000 images at 1024×1024 resolution.
    (Copied from PaperWithCode)

    The official way to prepare the dataset is to download img_celeba.7z from the original CelebA dataset and the delta
    files from the official GitHub repository. Then use dataset_tool.py to generate the high-quality images.

    However, I personally recommend downloading the CelebAMask-HQ dataset, which contains processed CelebA-HQ images.
    Note that the filenames in CelebAMask-HQ are sorted from 0 to 29999, which is inconsistent with the original CelebA
    filenames. A python script (`scripts/celebahq_map_filenames.py`) is provided to help convert the filenames.

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
