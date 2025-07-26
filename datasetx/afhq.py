import os
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .utils import extract_images


class AFHQ(Dataset):
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
            transform_fn: Optional[Callable] = None,
    ):
        if split not in ['train', 'test']:
            raise ValueError(f'Invalid split: {split}')
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform_fn = transform_fn

        # extract image paths
        image_root = os.path.join(self.root, split)
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')
        self.image_paths = extract_images(image_root)

        # extract labels
        self.labels = []
        for p in self.image_paths:
            self.labels.append(0 if 'cat' in p else 1 if 'dog' in p else 2)

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
