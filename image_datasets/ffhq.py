import os
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from .utils import extract_images


class FFHQ(Dataset):
    """The Flickr-Faces-HQ (FFHQ) Dataset.

    Please organize the dataset in the following file structure:

    root
    ├── ffhq-dataset-v2.json
    ├── LICENSE.txt
    ├── README.txt
    ├── thumbnails128x128
    │   ├── 00000.png
    │   ├── ...
    │   └── 69999.png
    └── images1024x1024
        ├── 00000.png
        ├── ...
        └── 69999.png

    Reference:
      - https://github.com/NVlabs/ffhq-dataset
      - https://paperswithcode.com/dataset/ffhq
      - https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            version: str = 'images1024x1024',
            transform_fn: Optional[Callable] = None,
    ):
        if split not in ['train', 'test', 'all']:
            raise ValueError(f'Invalid split: {split}')
        if version not in ['images1024x1024', 'thumbnails128x128']:
            raise ValueError(f'Invalid version: {version}')
        self.root = os.path.expanduser(root)
        self.split = split
        self.version = version
        self.transform_fn = transform_fn

        # extract image paths
        image_root = os.path.join(self.root, version)
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')
        self.image_paths = extract_images(image_root)
        if split == 'train':
            self.image_paths = list(filter(lambda p: 0 <= int(os.path.basename(p).split('.')[0]) < 60000, self.image_paths))
        elif split == 'test':
            self.image_paths = list(filter(lambda p: 60000 <= int(os.path.basename(p).split('.')[0]) < 70000, self.image_paths))

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
