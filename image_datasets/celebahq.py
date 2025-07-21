import os
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .utils import extract_images


class CelebAHQ(Dataset):
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
            transform_fn: Optional[Callable] = None,
    ):
        if split not in ['train', 'valid', 'test', 'all']:
            raise ValueError(f'Invalid split: {split}')
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform_fn = transform_fn

        # extract image paths
        image_root = os.path.join(self.root, 'CelebA-HQ-img')
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')

        def filter_func(p):
            if split == 'all':
                return True
            celeba_splits = [1, 162771, 182638, 202600]
            k = 0 if split == 'train' else (1 if split == 'valid' else 2)
            return celeba_splits[k] <= int(os.path.splitext(os.path.basename(p))[0]) < celeba_splits[k+1]

        self.image_paths = extract_images(image_root)
        self.image_paths = list(filter(filter_func, self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # read image
        x = Image.open(self.image_paths[index]).convert('RGB')
        x = to_tensor(x)
        sample = {'image': x}
        # apply transform
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample
