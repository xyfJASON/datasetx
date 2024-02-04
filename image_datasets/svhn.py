from typing import Optional, Callable

import torchvision.datasets
import torchvision.transforms as T
from torch.utils.data import Dataset


class SVHN(Dataset):
    """Extend torchvision.datasets.SVHN with one pre-defined transform.

    The pre-defined transform is:
      - 'resize' (default): Resize the image directly to the target size

    """

    def __init__(
            self,
            root: str,
            img_size: int,
            split: str = 'train',
            transform_type: Optional[str] = 'resize',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        if split not in ['train', 'test', 'extra']:
            raise ValueError(f'Invalid split: {split}')
        if transform_type not in ['resize', 'none'] and transform_type is not None:
            raise ValueError(f'Invalid transform_type: {transform_type}')

        self.img_size = img_size
        self.transform_type = transform_type
        if transform is None:
            transform = self.get_transform()

        self.svhn = torchvision.datasets.SVHN(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __len__(self):
        return len(self.svhn)

    def __getitem__(self, item):
        X, y = self.svhn[item]
        return X, y

    def get_transform(self):
        if self.transform_type == 'resize':
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'none' or self.transform_type is None:
            transform = None
        else:
            raise ValueError(f'Invalid transform_type: {self.transform_type}')
        return transform
