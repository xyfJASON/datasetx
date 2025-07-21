import os
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .utils import extract_images


class ImageNet(Dataset):
    """The ImageNet-1K (ILSVRC 2012) Dataset.

    Please organize the dataset in the following file structure:

    root
    ├── train
    │   ├── n01440764
    │   ├── ...
    │   └── n15075141
    ├── val
    │   ├── n01440764
    │   ├── ...
    │   └── n15075141
    └── test
        ├── ILSVRC2012_test_00000001.JPEG
        ├── ...
        └── ILSVRC2012_test_00100000.JPEG

    References:
      - https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
      - https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4
      - https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform_fn: Optional[Callable] = None,
    ):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Invalid split: {split}')
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform_fn = transform_fn

        # extract image paths
        image_root = os.path.join(self.root, split if split != 'valid' else 'val')
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')
        self.image_paths = extract_images(image_root)

        # extract class labels
        self.classes = None
        if self.split != 'test':
            class_names = [path.split('/')[-2] for path in self.image_paths]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            self.classes = [sorted_classes[x] for x in class_names]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # read image and label
        x = Image.open(self.image_paths[index]).convert('RGB')
        x = to_tensor(x)
        y = self.classes[index] if self.classes is not None else None
        sample = {'image': x, 'label': y}
        # apply transform
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample
