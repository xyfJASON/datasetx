import os
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .utils import extract_images


class Danbooru2019Portraits(Dataset):
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
            transform_fn: Optional[Callable] = None,
    ):
        self.root = os.path.expanduser(root)
        self.transform_fn = transform_fn

        # extract image paths
        image_root = os.path.join(self.root, 'portraits')
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')
        self.image_paths = extract_images(image_root)

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
