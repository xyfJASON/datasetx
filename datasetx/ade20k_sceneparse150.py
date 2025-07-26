import os
import numpy as np
from PIL import Image
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class ADE20KSceneParse150(Dataset):
    """The ADE20K SceneParse150 Dataset.

    Please organize the dataset in the following file structure:

    root
    ├── ADEChallengeData2016
    │   ├── annotations
    │   │   ├── training
    │   │   └── validation
    │   ├── images
    │   │   ├── training
    │   │   └── validation
    │   ├── objectInfo150.txt
    │   └── sceneCategories.txt
    └── release_test
        ├── list.txt
        ├── readme.txt
        └── testing

    References:
      - https://ade20k.csail.mit.edu/
      - http://sceneparsing.csail.mit.edu/
      - https://paperswithcode.com/dataset/ade20k

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform_fn: Optional[Callable] = None,
    ):
        self.root = os.path.expanduser(root)
        self.transform_fn = transform_fn
        if split in ['train', 'training']:
            self.images_root = os.path.join(self.root, 'ADEChallengeData2016', 'images', 'training')
            self.annotations_root = os.path.join(self.root, 'ADEChallengeData2016', 'annotations', 'training')
        elif split in ['val', 'valid', 'validation']:
            self.images_root = os.path.join(self.root, 'ADEChallengeData2016', 'images', 'validation')
            self.annotations_root = os.path.join(self.root, 'ADEChallengeData2016', 'annotations', 'validation')
        elif split in ['test', 'testing']:
            self.images_root = os.path.join(self.root, 'release_test', 'testing')
            self.annotations_root = None

        self.filenames = [f.replace('.jpg', '') for f in os.listdir(self.images_root) if f.endswith('.jpg')]
        self.filenames = list(sorted(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        # read image
        image_path = os.path.join(self.images_root, self.filenames[index] + '.jpg')
        image = Image.open(image_path).convert('RGB')
        # read annotation
        annotation = None
        if self.annotations_root is not None:
            annotation_path = os.path.join(self.annotations_root, self.filenames[index] + '.png')
            annotation = Image.open(annotation_path).convert('L')
        # convert to tensor
        image = to_tensor(image)
        if annotation is not None:
            annotation = torch.as_tensor(np.array(annotation), dtype=torch.int64)
        sample = {'image': image, 'annotation': annotation}
        # apply transform
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample
